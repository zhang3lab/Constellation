"""
Compare extracted absorbed-latent reference against FlashMLA latent output.
This is the reference-vs-kernel validation path.
"""
import argparse
import math
import numpy as np
import torch

import flash_mla

from server.deepseek_model_loader import DeepseekModelLoader
from server.test_utils import compare_arrays
from server.shallowmla_adapter import ShallowMLAAttentionWrapper
from server.absorbed_latent_ref import (
    build_ref_state_for_one_token,
    eager_absorbed_latent_attention,
)


class RefFlashCache:
    def __init__(self):
        self.cache_latent_list = []
        self.cache_k_rope_list = []

    def append_from_ref_state(self, ref_state):
        self.cache_latent_list.append(ref_state["cache_latent_1tok"].float())
        self.cache_k_rope_list.append(ref_state["cache_k_rope_1tok"].float())

    def materialize(self):
        cache_latent = torch.stack(self.cache_latent_list, dim=0)   # [T, kv]
        cache_k_rope = torch.stack(self.cache_k_rope_list, dim=0)   # [T, rope]
        return cache_latent, cache_k_rope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=str, required=True)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
    )
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

    # only used for freq_cis generation
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
    d = kv_lora_rank + qk_rope_head_dim   # absorbed Q/K dim for FlashMLA query/key
    dv = kv_lora_rank                     # latent output dim
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

    ref_cache = RefFlashCache()
    ys_ref = []
    ys_flash = []

    # prefill cache
    for t in range(args.prefill_len):
        x_t = x_all[:, t:t + 1, :]
        freq_t = shallow.freq_cis[t:t + 1].to(device=device, dtype=dtype)

        ref_state = build_ref_state_for_one_token(
            x_t,
            ws,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            freq_t=freq_t,
        )
        ref_cache.append_from_ref_state(ref_state)

        page = t // page_size
        off = t % page_size
        blocked_k[page, off, 0, :] = ref_state["blocked_k_token"][0, 0, :]
        cache_seqlens[0] = t + 1

    for i in range(args.decode_steps):
        pos = args.prefill_len + i
        x_step = x_all[:, pos:pos + 1, :]
        freq_t = shallow.freq_cis[pos:pos + 1].to(device=device, dtype=dtype)

        ref_state = build_ref_state_for_one_token(
            x_step,
            ws,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            freq_t=freq_t,
        )
        ref_cache.append_from_ref_state(ref_state)

        page = pos // page_size
        off = pos % page_size
        blocked_k[page, off, 0, :] = ref_state["blocked_k_token"][0, 0, :]
        cache_seqlens[0] = pos + 1

        cache_latent, cache_k_rope = ref_cache.materialize()

        y_ref = eager_absorbed_latent_attention(
            ref_state["q_nope_absorb"],
            ref_state["q_rope"],
            cache_latent,
            cache_k_rope,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        ).unsqueeze(1)  # [1,1,H,kv]
        ys_ref.append(y_ref)

        q_flash = ref_state["q_flash"].unsqueeze(1)  # [1,1,H,kv+rope]
        y_flash, _lse = flash_mla.flash_mla_with_kvcache(
            q_flash,
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
            f"decode_step_{i}_absorbed_ref_vs_flashmla",
            y_ref.detach().float().cpu().numpy(),
            y_flash.detach().float().cpu().numpy(),
        )

    y_ref_all = torch.cat(ys_ref, dim=1)
    y_flash_all = torch.cat(ys_flash, dim=1)

    compare_arrays(
        "decode_all_steps_absorbed_ref_vs_flashmla",
        y_ref_all.detach().float().cpu().numpy(),
        y_flash_all.detach().float().cpu().numpy(),
    )

    print("ALL ABSORBED REF VS FLASHMLA LATENT CASES PASSED")


if __name__ == "__main__":
    main()
