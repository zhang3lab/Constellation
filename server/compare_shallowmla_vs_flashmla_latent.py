import argparse
import math
import numpy as np
import torch

from server.deepseek_model_loader import DeepseekModelLoader
from server.test_utils import compare_arrays

import flash_mla


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
    ws: dict,
    *,
    num_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    freq_t: torch.Tensor,
):
    q_latent = x_token_norm @ ws["q_a_proj"].t()
    q_latent = rms_norm_t(q_latent, ws["q_a_layernorm"])
    q_full = q_latent @ ws["q_b_proj"].t()
    q_full = q_full.view(1, num_heads, qk_nope_head_dim + qk_rope_head_dim)

    q_nope = q_full[..., :qk_nope_head_dim]      # [1, H, 128]
    q_rope = q_full[..., qk_nope_head_dim:]      # [1, H, 64]
    q_rope = apply_rope_1tok(q_rope, freq_t)

    kv_down = x_token_norm @ ws["kv_a_proj_with_mqa"].t()
    kv_latent = kv_down[:, :kv_lora_rank]        # [1, 512]
    k_rope = kv_down[:, kv_lora_rank:]           # [1, 64]
    kv_latent_norm = rms_norm_t(kv_latent, ws["kv_a_layernorm"])
    k_rope = apply_rope_1tok(k_rope.view(1, 1, qk_rope_head_dim), freq_t).view(1, qk_rope_head_dim)

    kv_b = ws["kv_b_proj"].view(
        num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
    )
    kv_up_k = kv_b[:, :qk_nope_head_dim, :]      # [H, 128, 512]

    q_nope_absorb = torch.einsum("bhd,hdk->bhk", q_nope, kv_up_k)   # [1, H, 512]
    q_flash = torch.cat([q_nope_absorb, q_rope], dim=-1)            # [1, H, 576]

    # blocked_k entry: first 512 dims are latent, last 64 are rope
    blocked_k_token = torch.cat([kv_latent_norm, k_rope], dim=-1).view(1, 1, kv_lora_rank + qk_rope_head_dim)

    return q_nope_absorb, q_rope, q_flash, blocked_k_token


def eager_absorbed_latent_attention(
    q_nope_absorb: torch.Tensor,   # [1,H,512]
    q_rope: torch.Tensor,          # [1,H,64]
    cache_latent: torch.Tensor,    # [T,512]
    cache_k_rope: torch.Tensor,    # [T,64]
):
    # scores_latent: [1,H,T]
    scores_latent = torch.einsum("bhk,tk->bht", q_nope_absorb.float(), cache_latent.float())
    scores_rope = torch.einsum("bhr,tr->bht", q_rope.float(), cache_k_rope.float())
    scores = (scores_latent + scores_rope) / math.sqrt(q_nope_absorb.shape[-1] + q_rope.shape[-1])

    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)      # [1,H,T]
    out = torch.einsum("bht,tk->bhk", probs, cache_latent.float())   # [1,H,512]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=str, required=True)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loader = DeepseekModelLoader(args.model_root)
    mla_cfg = loader.mla_config()
    ws = {
        k: v.to(device=device, dtype=dtype)
        for k, v in loader.load_attention_block_weights_fp32(args.layer_id).items()
    }

    hidden_size = int(mla_cfg["dim"])
    num_heads = int(mla_cfg["num_heads"])
    kv_lora_rank = int(mla_cfg["kv_latent_rank"])
    qk_nope_head_dim = int(mla_cfg["qk_nrope_head_dim"])
    qk_rope_head_dim = int(mla_cfg["qk_rope_head_dim"])
    v_head_dim = int(mla_cfg["v_head_dim"])

    total_len = args.prefill_len + args.decode_steps
    x_all = torch.randn(1, total_len, hidden_size, device=device, dtype=dtype)

    # reuse ShallowMLA freq table logic
    from server.shallowmla_adapter import ShallowMLAAttentionWrapper
    shallow = ShallowMLAAttentionWrapper(
        model_loader=loader,
        layer_id=args.layer_id,
        max_batch_size=1,
        dtype=dtype,
        device="cuda",
        optim_type="triton",
    )

    page_size = args.page_size
    num_pages = math.ceil(total_len / page_size) + 4
    d = kv_lora_rank + qk_rope_head_dim  # 576
    dv = kv_lora_rank                    # 512
    h_q = num_heads
    h_kv = 1

    blocked_k = torch.full(
        (num_pages, page_size, h_kv, d),
        float("nan"),
        device=device,
        dtype=dtype,
    )

    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(1, -1)
    cache_seqlens = torch.zeros(1, dtype=torch.int32, device=device)

    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata()

    cache_latent_list = []
    cache_k_rope_list = []

    # prefill cache only
    for t in range(args.prefill_len):
        x_t = x_all[:, t:t+1, :]
        x_t_norm = rms_norm_t(x_t[:, 0, :], ws["input_layernorm"])
        freq_t = shallow.freq_cis[t:t+1].to(device=device, dtype=dtype)

        q_nope_absorb, q_rope, q_flash, blocked_k_token = build_one_token_q_and_cache_entry(
            x_t_norm,
            ws,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            freq_t=freq_t,
        )

        page = t // page_size
        off = t % page_size
        blocked_k[page, off, 0, :] = blocked_k_token[0, 0, :]
        cache_seqlens[0] = t + 1

        cache_latent_list.append(blocked_k_token[0, 0, :kv_lora_rank].float())
        cache_k_rope_list.append(blocked_k_token[0, 0, kv_lora_rank:].float())

    ys_flash = []
    ys_eager = []

    for i in range(args.decode_steps):
        pos = args.prefill_len + i
        x_step = x_all[:, pos:pos+1, :]
        x_step_norm = rms_norm_t(x_step[:, 0, :], ws["input_layernorm"])
        freq_t = shallow.freq_cis[pos:pos+1].to(device=device, dtype=dtype)

        q_nope_absorb, q_rope, q_flash, blocked_k_token = build_one_token_q_and_cache_entry(
            x_step_norm,
            ws,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            freq_t=freq_t,
        )

        # append current token to cache
        page = pos // page_size
        off = pos % page_size
        blocked_k[page, off, 0, :] = blocked_k_token[0, 0, :]
        cache_seqlens[0] = pos + 1

        cache_latent_list.append(blocked_k_token[0, 0, :kv_lora_rank].float())
        cache_k_rope_list.append(blocked_k_token[0, 0, kv_lora_rank:].float())

        cache_latent = torch.stack(cache_latent_list, dim=0)   # [T,512]
        cache_k_rope = torch.stack(cache_k_rope_list, dim=0)   # [T,64]

        # eager latent reference
        y_eager = eager_absorbed_latent_attention(
            q_nope_absorb,
            q_rope,
            cache_latent,
            cache_k_rope,
        )   # [1,H,512]
        y_eager = y_eager.unsqueeze(1)   # [1,1,H,512]
        ys_eager.append(y_eager)

        # flash mla
        q = q_flash.unsqueeze(1)         # [1,1,H,576]
        y_flash, lse = flash_mla.flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            dv,
            tile_scheduler_metadata,
            num_splits,
            causal=True,
        )
        ys_flash.append(y_flash)

        compare_arrays(
            f"decode_step_{i}_eager_absorbed_vs_flashmla",
            y_eager.detach().float().cpu().numpy(),
            y_flash.detach().float().cpu().numpy(),
        )

    y_eager_all = torch.cat(ys_eager, dim=1)
    y_flash_all = torch.cat(ys_flash, dim=1)

    compare_arrays(
        "decode_all_steps_eager_absorbed_vs_flashmla",
        y_eager_all.detach().float().cpu().numpy(),
        y_flash_all.detach().float().cpu().numpy(),
    )

    print("ALL LATENT CASES PASSED")


if __name__ == "__main__":
    main()
