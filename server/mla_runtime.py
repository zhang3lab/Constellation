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
            freq_cis: [seq_len, qk_rope_head_dim/2, 2] or ShallowMLA-compatible slice
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
     
        q_latent = torch.matmul(x, weights["q_a_proj"].t())
        q_latent = fused_rms_norm(
            q_latent,
            (q_latent.shape[-1],),
            weights["q_a_layernorm"],
            self.eps,
        )
        q = torch.matmul(q_latent, weights["q_b_proj"].t())
     
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nrope, q_rope = q.split(
            [self.qk_nrope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_rope = fused_apply_rotary_emb(q_rope, freq_cis)
     
        kv_down = torch.matmul(x, weights["kv_a_proj_with_mqa"].t())
        kv_latent, k_rope = kv_down.split(
            [self.kv_latent_rank, self.qk_rope_head_dim], dim=-1
        )
        k_rope = fused_apply_rotary_emb(k_rope.unsqueeze(2), freq_cis).squeeze(2)
     
        normalized_kv_latent = fused_rms_norm(
            kv_latent,
            (kv_latent.shape[-1],),
            weights["kv_a_layernorm"],
            self.eps,
        )
     
        end_pos = start_pos + seq_len
     
        for b_idx in range(batch_size):
            cache_manager.update(
                batch_idx=b_idx,
                start_pos=start_pos,
                kv_latent=normalized_kv_latent[b_idx],
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
     
        proj_kv_up_weight = weights["kv_b_proj"].view(
            self.num_heads,
            self.qk_nrope_head_dim + self.v_head_dim,
            self.kv_latent_rank,
        )
        proj_kv_up_weight_q_nrope_absorbed = proj_kv_up_weight[:, : self.qk_nrope_head_dim, :]
        proj_kv_up_weight_v = proj_kv_up_weight[:, -self.v_head_dim :, :]
     
        q_nrope_absorb = torch.einsum(
            "blhd,hdk->blhk", q_nrope, proj_kv_up_weight_q_nrope_absorbed
        )
        q_flash = torch.cat([q_nrope_absorb, q_rope], dim=-1)
        blocked_k_token = torch.cat([normalized_kv_latent, k_rope], dim=-1)
     
        kernel_dtype = tl.float16 if x.dtype == torch.float16 else tl.float32
        scores = fused_qk_attention(
            q_nrope_absorb,
            q_rope,
            stacked_kv_latent,
            stacked_k_rope,
            self.softmax_scale,
            kernel_version=2,
            dtype=kernel_dtype,
        )
     
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(0)
            fused_mask_softmax(scores, mask)
        else:
            scores = scores.softmax(dim=-1)
     
        x = torch.einsum("blht,btk->blhk", scores, stacked_kv_latent)
        x = torch.einsum("blhk,hdk->blhd", x, proj_kv_up_weight_v)
        x = x.flatten(start_dim=2)
        x = torch.matmul(x, weights["o_proj"].t())
     
        if not return_aux:
            return x
     
        aux = {
            "q_flash": q_flash.detach().float().cpu().numpy(),
            "blocked_k_token": blocked_k_token.detach().float().cpu().numpy(),
        }
        return x, aux
