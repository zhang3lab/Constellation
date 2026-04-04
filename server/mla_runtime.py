from __future__ import annotations

from typing import Optional

import torch
import triton.language as tl

from third_party.ShallowMLA.mla import (
    fused_apply_rotary_emb,
    fused_mask_softmax,
    fused_qk_attention,
    fused_rms_norm,
)

from server.debug_utils import DebugTensorCollector


def apply_rotary_emb_hf_exact_torch(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    Match HF DeepSeek apply_rotary_pos_emb semantics exactly.

    Args:
        x: [B, L, H, D] in interleaved layout
           [a0, b0, a1, b1, ...]
        freqs_cis: [L, D // 2, 2]
           freqs_cis[..., 0] = cos
           freqs_cis[..., 1] = sin

    Returns:
        [B, L, H, D] in HF post-rotary layout
    """
    if x.ndim != 4:
        raise ValueError(f"x expected [B,L,H,D], got {tuple(x.shape)}")
    if freqs_cis.ndim != 3:
        raise ValueError(f"freqs_cis expected [L,D//2,2], got {tuple(freqs_cis.shape)}")

    B, L, H, D = x.shape
    if D % 2 != 0:
        raise ValueError(f"rotary dim must be even, got D={D}")
    d_half = D // 2

    if int(freqs_cis.shape[0]) != L:
        raise ValueError(
            f"freqs_cis seq mismatch: x has L={L}, freqs_cis has {int(freqs_cis.shape[0])}"
        )
    if int(freqs_cis.shape[1]) != d_half:
        raise ValueError(
            f"freqs_cis dim mismatch: x has D/2={d_half}, freqs_cis has {int(freqs_cis.shape[1])}"
        )
    if int(freqs_cis.shape[2]) != 2:
        raise ValueError(f"freqs_cis last dim must be 2, got {int(freqs_cis.shape[2])}")

    dtype = x.dtype

    # HF:
    # q = q.view(..., d//2, 2).transpose(...).reshape(...)
    # This converts interleaved -> half-split.
    x = x.view(B, L, H, d_half, 2).transpose(4, 3).reshape(B, L, H, D).float()

    cos_half = freqs_cis[..., 0].to(device=x.device, dtype=torch.float32)  # [L, D/2]
    sin_half = freqs_cis[..., 1].to(device=x.device, dtype=torch.float32)  # [L, D/2]

    # HF full-dim half-split broadcast layout:
    # [c0..c_{d/2-1}, c0..c_{d/2-1}]
    cos = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0).unsqueeze(2)  # [1, L, 1, D]
    sin = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0).unsqueeze(2)  # [1, L, 1, D]

    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    rotate_half_x = torch.cat([-x2, x1], dim=-1)

    out = (x * cos) + (rotate_half_x * sin)
    return out.to(dtype=dtype)


class MLARuntime:
    def __init__(
        self,
        *,
        dim: int,
        kv_latent_rank: int,
        q_latent_rank: int,
        num_heads: int,
        qk_nrope_head_dim: int,
        v_head_dim: int,
        qk_rope_head_dim: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-6,
    ):
        self.dim = int(dim)
        self.kv_latent_rank = int(kv_latent_rank)
        self.q_latent_rank = int(q_latent_rank)
        self.num_heads = int(num_heads)
        self.qk_nrope_head_dim = int(qk_nrope_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.qk_head_dim = self.qk_nrope_head_dim + self.qk_rope_head_dim
        self.absorbed_qk_head_dim = self.kv_latent_rank + self.qk_rope_head_dim
        self.softmax_scale = 1.0 / (self.absorbed_qk_head_dim ** 0.5)
        self.dtype = dtype
        self.eps = float(eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        start_pos: int,
        freq_cis: torch.Tensor,
        weights: dict[str, torch.Tensor],
        cache_manager,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """
        Args:
            x: [batch_size, seq_len, dim]
            start_pos: logical start position for this chunk
            freq_cis: [seq_len, qk_rope_head_dim/2, 2]
            weights:
                {
                    "q_a_proj",
                    "q_a_layernorm",
                    "q_b_proj",
                    "kv_a_proj_with_mqa",
                    "kv_a_layernorm",
                    "kv_b_proj",
                    "o_proj",
                }
            cache_manager: external PageAttentionCacheManager
            mask: optional [seq_len_q, seq_len_k]
            return_aux: whether to return intermediate tensors for compare/debug
     
        Returns:
            if return_aux is False:
                y: [batch_size, seq_len, dim]
            else:
                (y, aux_dict)
        """
        if cache_manager is None:
            raise RuntimeError("MLARuntime requires external cache_manager")
     
        if x.device.type != "cuda":
            raise RuntimeError(f"MLARuntime expects CUDA tensor input, got {x.device}")
        torch.cuda.set_device(x.device)
     
        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise RuntimeError(f"x last dim mismatch: got={dim} expected={self.dim}")
     
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
     
        dbg = DebugTensorCollector(enabled=return_aux)
     
        x_norm = x
     
        x = torch.matmul(x_norm, weights["q_a_proj"].t())
        dbg.add("q_latent_pre_norm", x)
     
        x = fused_rms_norm(
            x,
            (x.shape[-1],),
            weights["q_a_layernorm"],
            self.eps,
        )
        dbg.add("q_latent_post_norm", x)
     
        x = torch.matmul(x, weights["q_b_proj"].t())
        dbg.add("q_pre_split", x)
     
        x = x.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nrope, q_rope = x.split(
            [self.qk_nrope_head_dim, self.qk_rope_head_dim], dim=-1
        )
     
        dbg.add("q_rope_pre_rotary", q_rope)
        q_rope = apply_rotary_emb_hf_exact_torch(q_rope, freq_cis)
        dbg.add("q_rope_post_rotary", q_rope)
     
        x = torch.matmul(x_norm, weights["kv_a_proj_with_mqa"].t())
        kv_latent, k_rope = x.split(
            [self.kv_latent_rank, self.qk_rope_head_dim], dim=-1
        )
        dbg.add("cache_latent_raw", kv_latent)
        dbg.add("cache_k_rope", k_rope)
     
        k_rope = apply_rotary_emb_hf_exact_torch(k_rope.unsqueeze(2), freq_cis).squeeze(2)
     
        kv_latent = fused_rms_norm(
            kv_latent,
            (kv_latent.shape[-1],),
            weights["kv_a_layernorm"],
            self.eps,
        )
        dbg.add("cache_latent", kv_latent)
     
        end_pos = start_pos + seq_len
        dbg.add_meta(
            kind="attention_block",
            impl="mla_runtime_triton",
            device=str(x_norm.device),
            start_pos=int(start_pos),
            seq_len=int(seq_len),
            end_pos=int(end_pos),
        )
        dbg.add("freq_cis", freq_cis)
     
        for b_idx in range(batch_size):
            cache_manager.update(
                batch_idx=b_idx,
                start_pos=start_pos,
                kv_latent=kv_latent[b_idx],
                k_rope=k_rope[b_idx],
            )
     
        all_kv_latent = []
        all_k_rope = []
        for b_idx in range(batch_size):
            batch_kv_latent, batch_k_rope = cache_manager.retrieve(
                batch_idx=b_idx,
                start_pos=0,
                end_pos=end_pos,
            )
            all_kv_latent.append(batch_kv_latent)
            all_k_rope.append(batch_k_rope)
     
        stacked_kv_latent = torch.stack(all_kv_latent, dim=0)
        stacked_k_rope = torch.stack(all_k_rope, dim=0)
     
        x = weights["kv_b_proj"].view(
            self.num_heads,
            self.qk_nrope_head_dim + self.v_head_dim,
            self.kv_latent_rank,
        )
        q_absorb_w = x[:, : self.qk_nrope_head_dim, :]
        v_w = x[:, -self.v_head_dim :, :]
     
        x = torch.einsum("blhd,hdk->blhk", q_nrope, q_absorb_w)
        dbg.add("q_nope_absorb", x)
     
        q_flash = torch.cat([x, q_rope], dim=-1)
        dbg.add("q_flash", q_flash)
     
        blocked_k_token = torch.cat([kv_latent, k_rope], dim=-1)
        dbg.add("blocked_k_token", blocked_k_token)
     
        kernel_dtype = tl.float16 if x_norm.dtype == torch.float16 else tl.float32
        x = fused_qk_attention(
            x,
            q_rope,
            stacked_kv_latent,
            stacked_k_rope,
            self.softmax_scale,
            kernel_version=2,
            dtype=kernel_dtype,
        )
        dbg.add("scores_pre_softmax", x)
     
        if return_aux:
            # optional: pure nope-only score for exact HF compare
            scores_nope_only = torch.einsum("blhk,btk->blht", q_nrope_absorb, stacked_kv_latent)
            dbg.add("scores_nope_only", scores_nope_only)
        print("[MLA] using causal mask", mask is None, start_pos, seq_len, end_pos)
        if mask is None:
            q_pos = torch.arange(start_pos, end_pos, device=x.device)
            k_pos = torch.arange(0, end_pos, device=x.device)
            allowed = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
     
            mask = torch.zeros(
                (1, seq_len, 1, end_pos),
                device=x.device,
                dtype=x.dtype,
            )
            mask = mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(2), -1e9)
        else:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(2)
            elif mask.ndim == 4:
                pass
            else:
                raise RuntimeError(f"unsupported attention mask shape: {tuple(mask.shape)}")
     
            if mask.dtype != x.dtype:
                mask = mask.to(dtype=x.dtype)
            if mask.device != x.device:
                mask = mask.to(device=x.device)
     
        print("[MLA] mask stats", mask.shape, mask.dtype, float(mask.min().item()), float(mask.max().item()))
        fused_mask_softmax(x, mask)
        dbg.add("scores_post_softmax", x)
     
        x = torch.einsum("blht,btk->blhk", x, stacked_kv_latent)
        x = torch.einsum("blhk,hdk->blhd", x, v_w)
        dbg.add("last_value_heads", x)
     
        x = x.flatten(start_dim=2)
        x = torch.matmul(x, weights["o_proj"].t())
     
        if not return_aux:
            return x
     
        return x, dbg.export()
