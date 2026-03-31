import math
from typing import Any, Dict, Tuple

import numpy as np
import torch

from server.full_model_ref import ModelExecResult


def rms_norm_t(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


def apply_rope_1tok(x: torch.Tensor, freq_cis_1tok: torch.Tensor) -> torch.Tensor:
    x2 = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    xr = x2[..., 0]
    xi = x2[..., 1]
    cr = freq_cis_1tok[..., 0]
    ci = freq_cis_1tok[..., 1]
    yr = xr * cr - xi * ci
    yi = xr * ci + xi * cr
    y = torch.stack([yr, yi], dim=-1).reshape(*x.shape[:-1], x.shape[-1])
    return y.to(x.dtype)


def build_one_token_q_and_cache_entry(
    x_token_norm: torch.Tensor,
    ws: Dict[str, torch.Tensor],
    *,
    num_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    freq_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      q_nope_absorb: [1, H, kv_lora_rank]
      q_rope:        [1, H, qk_rope_head_dim]
      q_flash:       [1, H, kv_lora_rank + qk_rope_head_dim]
      blocked_k:     [1, 1, kv_lora_rank + qk_rope_head_dim]
    """
    q_latent = x_token_norm @ ws["q_a_proj"].t()
    q_latent = rms_norm_t(q_latent, ws["q_a_layernorm"])
    q_full = q_latent @ ws["q_b_proj"].t()
    q_full = q_full.view(1, num_heads, qk_nope_head_dim + qk_rope_head_dim)

    q_nope = q_full[..., :qk_nope_head_dim]
    q_rope = q_full[..., qk_nope_head_dim:]
    q_rope = apply_rope_1tok(q_rope, freq_t)

    kv_down = x_token_norm @ ws["kv_a_proj_with_mqa"].t()
    kv_latent = kv_down[:, :kv_lora_rank]
    k_rope = kv_down[:, kv_lora_rank:]
    kv_latent_norm = rms_norm_t(kv_latent, ws["kv_a_layernorm"])
    k_rope = apply_rope_1tok(
        k_rope.view(1, 1, qk_rope_head_dim), freq_t
    ).view(1, qk_rope_head_dim)

    kv_b = ws["kv_b_proj"].view(
        num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
    )
    kv_up_k = kv_b[:, :qk_nope_head_dim, :]

    q_nope_absorb = torch.einsum("bhd,hdk->bhk", q_nope, kv_up_k)
    q_flash = torch.cat([q_nope_absorb, q_rope], dim=-1)
    blocked_k = torch.cat([kv_latent_norm, k_rope], dim=-1).view(
        1, 1, kv_lora_rank + qk_rope_head_dim
    )
    return q_nope_absorb, q_rope, q_flash, blocked_k


def eager_absorbed_latent_attention(
    q_nope_absorb: torch.Tensor,
    q_rope: torch.Tensor,
    cache_latent: torch.Tensor,
    cache_k_rope: torch.Tensor,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    """
    Args:
      q_nope_absorb: [1, H, kv_lora_rank]
      q_rope:        [1, H, qk_rope_head_dim]
      cache_latent:  [T, kv_lora_rank]
      cache_k_rope:  [T, qk_rope_head_dim]

    Returns:
      out:           [1, H, kv_lora_rank]
    """
    scores_latent = torch.einsum(
        "bhk,tk->bht", q_nope_absorb.float(), cache_latent.float()
    )
    scores_rope = torch.einsum(
        "bhr,tr->bht", q_rope.float(), cache_k_rope.float()
    )
    scores = (scores_latent + scores_rope) / math.sqrt(
        kv_lora_rank + qk_rope_head_dim
    )

    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.einsum("bht,tk->bhk", probs, cache_latent.float())
    return out.to(q_nope_absorb.dtype)


def split_blocked_k(
    blocked_k_token: torch.Tensor,
    *,
    kv_lora_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      blocked_k_token: [1,1,kv_lora_rank + qk_rope_head_dim] or [kv_lora_rank + qk_rope_head_dim]

    Returns:
      cache_latent_1tok: [kv_lora_rank]
      cache_k_rope_1tok: [qk_rope_head_dim]
    """
    x = blocked_k_token
    if x.dim() == 3:
        x = x[0, 0]
    return x[:kv_lora_rank], x[kv_lora_rank:]


def latent_to_final_hidden(
    latent_out: torch.Tensor,
    ws: Dict[str, torch.Tensor],
    *,
    num_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> torch.Tensor:
    """
    latent_out: [1, H, kv_lora_rank]
    return:     [1, hidden_size]
    """
    kv_b = ws["kv_b_proj"].view(
        num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
    )
    kv_up_v = kv_b[:, qk_nope_head_dim:, :]
    value_heads = torch.einsum("bhk,hvk->bhv", latent_out, kv_up_v)
    value_flat = value_heads.reshape(1, num_heads * v_head_dim)
    return value_flat @ ws["o_proj"].t()


def build_ref_state_for_one_token(
    x_token: torch.Tensor,
    ws: Dict[str, torch.Tensor],
    *,
    num_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    freq_t: torch.Tensor,
):
    """
    Convenience helper used by higher-level compare scripts.

    Args:
      x_token: [1,1,hidden]

    Returns a dict with:
      x_token_norm
      q_nope_absorb
      q_rope
      q_flash
      blocked_k_token
      cache_latent_1tok
      cache_k_rope_1tok
    """
    x_token_norm = rms_norm_t(x_token[:, 0, :], ws["input_layernorm"])
    q_nope_absorb, q_rope, q_flash, blocked_k_token = build_one_token_q_and_cache_entry(
        x_token_norm,
        ws,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        freq_t=freq_t,
    )
    cache_latent_1tok, cache_k_rope_1tok = split_blocked_k(
        blocked_k_token,
        kv_lora_rank=kv_lora_rank,
    )
    return {
        "x_token_norm": x_token_norm,
        "q_nope_absorb": q_nope_absorb,
        "q_rope": q_rope,
        "q_flash": q_flash,
        "blocked_k_token": blocked_k_token,
        "cache_latent_1tok": cache_latent_1tok,
        "cache_k_rope_1tok": cache_k_rope_1tok,
    }


def _cache_manager_to_dense_cache(
    cache_manager,
    *,
    batch_idx: int,
    end_pos: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_latent, cache_k_rope = cache_manager.retrieve(
        batch_idx=batch_idx,
        start_pos=0,
        end_pos=end_pos,
    )
    return (
        cache_latent.to(device=device, dtype=dtype),
        cache_k_rope.to(device=device, dtype=dtype),
    )


def run_attention_block_ref(
    session,
    hidden_in: np.ndarray,
    layer_id: int,
    *,
    position_ids=None,
    attention_mask=None,
    kv_cache=None,
    return_aux: bool = False,
) -> ModelExecResult:
    del attention_mask  # single-token causal path for now

    if session.backbone_store is None:
        raise RuntimeError("session.backbone_store is not initialized")
    if session.freq_cis_by_device is None:
        raise RuntimeError("session.freq_cis_by_device is not initialized")
    if kv_cache is None:
        raise RuntimeError("kv_cache is required for run_attention_block_ref")

    layer_id = int(layer_id)
    layer_entry = session.backbone_store.layer(layer_id)
    ws = layer_entry["attention"]
    dev = str(layer_entry["device"])
    runtime_dtype = session.backbone_store.dtype
    mla_cfg = session.backbone_store.mla_cfg

    x = torch.from_numpy(
        np.asarray(hidden_in, dtype=np.float32).reshape(-1)
    ).to(device=dev, dtype=runtime_dtype).view(1, 1, -1)

    start_pos = 0 if position_ids is None else int(np.asarray(position_ids).reshape(-1)[0])
    freq_t = session.freq_cis_by_device[dev][start_pos : start_pos + 1]

    state = build_ref_state_for_one_token(
        x,
        ws,
        num_heads=int(mla_cfg["num_heads"]),
        kv_lora_rank=int(mla_cfg["kv_latent_rank"]),
        qk_nope_head_dim=int(mla_cfg["qk_nrope_head_dim"]),
        qk_rope_head_dim=int(mla_cfg["qk_rope_head_dim"]),
        v_head_dim=int(mla_cfg["v_head_dim"]),
        freq_t=freq_t,
    )

    cache_manager = kv_cache[layer_id]
    cache_manager.update(
        batch_idx=0,
        start_pos=start_pos,
        kv_latent=state["cache_latent_1tok"].view(1, -1),
        k_rope=state["cache_k_rope_1tok"].view(1, -1),
    )

    cache_latent, cache_k_rope = _cache_manager_to_dense_cache(
        cache_manager,
        batch_idx=0,
        end_pos=start_pos + 1,
        device=dev,
        dtype=runtime_dtype,
    )

    latent_out = eager_absorbed_latent_attention(
        state["q_nope_absorb"],
        state["q_rope"],
        cache_latent,
        cache_k_rope,
        kv_lora_rank=int(mla_cfg["kv_latent_rank"]),
        qk_rope_head_dim=int(mla_cfg["qk_rope_head_dim"]),
    )

    out = latent_to_final_hidden(
        latent_out,
        ws,
        num_heads=int(mla_cfg["num_heads"]),
        kv_lora_rank=int(mla_cfg["kv_latent_rank"]),
        qk_nope_head_dim=int(mla_cfg["qk_nrope_head_dim"]),
        v_head_dim=int(mla_cfg["v_head_dim"]),
    )

    out_np = out[0].detach().float().cpu().numpy().astype(np.float32, copy=False)

    aux: dict[str, Any] = {}
    if return_aux:
        aux = {
            "kind": "attention_block",
            "impl": "absorbed_latent_ref",
            "layer_id": layer_id,
            "device": dev,
            "start_pos": start_pos,
            "q_flash": state["q_flash"].detach().cpu().numpy(),
            "blocked_k_token": state["blocked_k_token"].detach().cpu().numpy(),
        }

    return ModelExecResult(output=out_np, aux=aux)
