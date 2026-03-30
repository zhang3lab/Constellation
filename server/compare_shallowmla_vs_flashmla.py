import argparse
import math
import numpy as np
import torch

from server.deepseek_model_loader import DeepseekModelLoader
from server.shallowmla_adapter import ShallowMLAAttentionWrapper
from server.test_utils import compare_arrays

from flash_mla import get_mla_metadata, flash_mla_with_kvcache


def rms_norm_t(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


def apply_rope_1tok(x: torch.Tensor, freq_cis_1tok: torch.Tensor) -> torch.Tensor:
    # x: [..., 64], freq_cis_1tok: [1, 32, 2]
    x2 = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    xr = x2[..., 0]
    xi = x2[..., 1]
    cr = freq_cis_1tok[..., 0]
    ci = freq_cis_1tok[..., 1]
    yr = xr * cr - xi * ci
    yi = xr * ci + xi * cr
    y = torch.stack([yr, yi], dim=-1).reshape(*x.shape[:-1], x.shape[-1])
    return y.to(x.dtype)


def build_flashmla_inputs_for_one_token(
    x_token_norm: torch.Tensor,
    ws: dict,
    *,
    num_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
):
    # x_token_norm: [1, hidden]
    q_latent = x_token_norm @ ws["q_a_proj"].t()                       # [1, q_rank]
    q_latent = rms_norm_t(q_latent, ws["q_a_layernorm"])
    q_full = q_latent @ ws["q_b_proj"].t()                            # [1, H * 192]
    q_full = q_full.view(1, num_heads, qk_nope_head_dim + qk_rope_head_dim)

    q_nope = q_full[..., :qk_nope_head_dim]                           # [1, H, 128]
    q_rope = q_full[..., qk_nope_head_dim:]                           # [1, H, 64]

    kv_down = x_token_norm @ ws["kv_a_proj_with_mqa"].t()             # [1, 512+64]
    kv_latent = kv_down[:, :kv_lora_rank]                             # [1, 512]
    k_rope = kv_down[:, kv_lora_rank:]                                # [1, 64]
    kv_latent_norm = rms_norm_t(kv_latent, ws["kv_a_layernorm"])      # [1, 512]

    kv_b = ws["kv_b_proj"].view(
        num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
    )
    kv_up_k = kv_b[:, :qk_nope_head_dim, :]                           # [H, 128, 512]
    kv_up_v = kv_b[:, qk_nope_head_dim:, :]                           # [H, 128, 512]

    q_nope_absorb = torch.einsum("bhd,hdk->bhk", q_nope, kv_up_k)     # [1, H, 512]

    # FlashMLA dense decode MQA mode: d_qk = 576 = 512 latent + 64 rope. README support matrix mentions MQA mode with head_dim_k=576 and head_dim_v=512 for MLA mode. [oai_citation:2‡GitHub](https://github.com/deepseek-ai/FlashMLA)
    q_flash = torch.cat([q_nope_absorb, q_rope], dim=-1)              # [1, H, 576]

    # one KV head in MQA mode
    kv_token = torch.cat([kv_latent_norm, k_rope], dim=-1).view(1, 1, kv_lora_rank + qk_rope_head_dim)  # [1,1,576]

    # later: latent output [B,1,H,512] -> V-up -> O-proj
    value_up_weight = kv_up_v.transpose(1, 2).contiguous()            # [H, 512, 128]

    return q_flash, kv_token, value_up_weight


def build_block_table(batch_size: int, num_pages: int, device: torch.device) -> torch.Tensor:
    if batch_size != 1:
        raise RuntimeError("this script currently assumes batch_size=1")
    return torch.arange(num_pages, device=device, dtype=torch.int32).view(1, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=str, required=True)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loader = DeepseekModelLoader(args.model_root)
    mla_cfg = loader.mla_config()

    hidden_size = int(mla_cfg["dim"])
    num_heads = int(mla_cfg["num_heads"])
    kv_lora_rank = int(mla_cfg["kv_latent_rank"])
    qk_nope_head_dim = int(mla_cfg["qk_nrope_head_dim"])
    qk_rope_head_dim = int(mla_cfg["qk_rope_head_dim"])
    v_head_dim = int(mla_cfg["v_head_dim"])

    total_len = args.prefill_len + args.decode_steps
    x_all = torch.randn(1, total_len, hidden_size, device=device, dtype=dtype)

    shallow = ShallowMLAAttentionWrapper(
        model_loader=loader,
        layer_id=args.layer_id,
        max_batch_size=1,
        dtype=dtype,
        device="cuda",
        optim_type="triton",
    )

    ws = {
        k: v.to(device=device, dtype=dtype)
        for k, v in loader.load_attention_block_weights_fp32(args.layer_id).items()
    }

    page_size = args.page_size
    num_pages = math.ceil(total_len / page_size) + 4
    d_qk = kv_lora_rank + qk_rope_head_dim  # 576
    d_v = kv_lora_rank                      # 512 latent output
    h_kv = 1

    # KV cache layout here follows the README's dense MLA decode calling convention.
    # The exact packed layout may need one small adjustment against the repo test on your machine.
    kvcache = torch.zeros(num_pages, page_size, h_kv, d_qk, device=device, dtype=dtype)
    block_table = build_block_table(1, num_pages, device)
    cache_seqlens = torch.zeros(1, device=device, dtype=torch.int32)

    def recompute_metadata():
        return get_mla_metadata(
            cache_seqlens,
            1 * num_heads // h_kv,   # s_q * h_q // h_kv
            h_kv,
            num_heads,
            False,                   # is_fp8
            0,                       # topk for dense decode
        )

    # prefill both sides
    with torch.no_grad():
        _ = shallow.forward(x_all[:, :args.prefill_len, :], start_pos=0, mask=None)

    for t in range(args.prefill_len):
        x_t = x_all[:, t:t+1, :]
        x_t_norm = rms_norm_t(x_t[:, 0, :], ws["input_layernorm"])
        q_flash, kv_token, value_up_weight = build_flashmla_inputs_for_one_token(
            x_t_norm,
            ws,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
        )

        freq_t = shallow.freq_cis[t:t+1].to(device=device, dtype=dtype)
        q_flash[..., kv_lora_rank:] = apply_rope_1tok(q_flash[..., kv_lora_rank:], freq_t)
        kv_token[..., kv_lora_rank:] = apply_rope_1tok(kv_token[..., kv_lora_rank:], freq_t)

        page = t // page_size
        off = t % page_size
        kvcache[page, off, 0, :] = kv_token[0, 0, :]
        cache_seqlens[0] = t + 1

    ys_shallow = []
    ys_flash = []

    for i in range(args.decode_steps):
        pos = args.prefill_len + i
        x_step = x_all[:, pos:pos+1, :]

        with torch.no_grad():
            y_shallow = shallow.forward(x_step, start_pos=pos, mask=None)
        ys_shallow.append(y_shallow)

        x_step_norm = rms_norm_t(x_step[:, 0, :], ws["input_layernorm"])
        q_flash, kv_token, value_up_weight = build_flashmla_inputs_for_one_token(
            x_step_norm,
            ws,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
        )

        freq_t = shallow.freq_cis[pos:pos+1].to(device=device, dtype=dtype)
        q_flash[..., kv_lora_rank:] = apply_rope_1tok(q_flash[..., kv_lora_rank:], freq_t)
        kv_token[..., kv_lora_rank:] = apply_rope_1tok(kv_token[..., kv_lora_rank:], freq_t)

        page = pos // page_size
        off = pos % page_size
        kvcache[page, off, 0, :] = kv_token[0, 0, :]
        cache_seqlens[0] = pos + 1

        tile_scheduler_metadata, num_splits = recompute_metadata()

        q_i = q_flash.unsqueeze(1)  # [1,1,Hq,576]

        out_latent, lse = flash_mla_with_kvcache(
            q_i,
            kvcache,
            block_table,
            cache_seqlens,
            d_v,
            tile_scheduler_metadata,
            num_splits,
            True,     # is_causal
            False,    # is_fp8_kvcache
            None,     # indices for sparse decode; None for dense
        )

        out_v = torch.einsum("bshk,hkd->bshd", out_latent, value_up_weight)
        out_v = out_v.reshape(1, 1, num_heads * v_head_dim)
        y_flash = out_v @ ws["o_proj"].t()
        ys_flash.append(y_flash)

        compare_arrays(
            f"decode_step_{i}_shallow_triton_vs_flashmla",
            y_shallow.detach().float().cpu().numpy(),
            y_flash.detach().float().cpu().numpy(),
        )

    y_shallow_all = torch.cat(ys_shallow, dim=1)
    y_flash_all = torch.cat(ys_flash, dim=1)

    compare_arrays(
        "decode_all_steps_shallow_triton_vs_flashmla",
        y_shallow_all.detach().float().cpu().numpy(),
        y_flash_all.detach().float().cpu().numpy(),
    )

    print("ALL SHALLOWMLA vs FLASHMLA CASES PASSED")


if __name__ == "__main__":
    main()
