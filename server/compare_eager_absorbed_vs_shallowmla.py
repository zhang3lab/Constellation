import argparse
import numpy as np
import torch

from server.deepseek_model_loader import DeepseekModelLoader
from server.test_utils import compare_arrays
from server.shallowmla_adapter import ShallowMLAAttentionWrapper
from server.absorbed_latent_ref import (
    build_ref_state_for_one_token,
    eager_absorbed_latent_attention,
    latent_to_final_hidden,
)


class RefAbsorbedLatentCache:
    def __init__(self):
        self.cache_latent_list = []
        self.cache_k_rope_list = []

    def append_from_ref_state(self, ref_state):
        self.cache_latent_list.append(ref_state["cache_latent_1tok"].float())
        self.cache_k_rope_list.append(ref_state["cache_k_rope_1tok"].float())

    def materialize(self):
        cache_latent = torch.stack(self.cache_latent_list, dim=0)
        cache_k_rope = torch.stack(self.cache_k_rope_list, dim=0)
        return cache_latent, cache_k_rope


def compare_one_debug_tensor(name, a, b):
    compare_arrays(
        name,
        a.detach().float().cpu().numpy(),
        b.detach().float().cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=str, required=True)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optim-type",
        type=str,
        default="triton",
        choices=["torch", "triton"],
    )
    parser.add_argument(
        "--debug-first-step",
        action="store_true",
        help="compare intermediate tensors for decode step 0",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

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

    shallow = ShallowMLAAttentionWrapper(
        model_loader=loader,
        layer_id=args.layer_id,
        max_batch_size=1,
        dtype=dtype,
        device="cuda",
        optim_type=args.optim_type,
    )

    ref_cache = RefAbsorbedLatentCache()

    # prefill: build ref cache and shallow cache
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

        _ = shallow.forward_debug(x_t, start_pos=t)

    ys_ref_latent = []
    ys_shallow_latent = []
    ys_ref_hidden = []
    ys_shallow_hidden = []

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
        cache_latent, cache_k_rope = ref_cache.materialize()

        y_ref_latent = eager_absorbed_latent_attention(
            ref_state["q_nope_absorb"],
            ref_state["q_rope"],
            cache_latent,
            cache_k_rope,
        ).unsqueeze(1)  # [1,1,H,kv]

        y_ref_hidden = latent_to_final_hidden(
            y_ref_latent[:, 0],
            ws,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
        ).unsqueeze(1)  # [1,1,hidden]

        out_shallow = shallow.forward_debug(x_step, start_pos=pos)
        y_shallow_latent = out_shallow["latent"]  # [1,1,H,kv]
        y_shallow_hidden = out_shallow["hidden"]  # [1,1,hidden]

        if args.debug_first_step and i == 0:
            compare_one_debug_tensor(
                "debug_step0_q_nrope_absorb_ref_vs_shallow",
                ref_state["q_nope_absorb"].unsqueeze(1),   # [1,1,H,kv]
                out_shallow["q_nrope_absorb"],             # [1,1,H,kv]
            )
            compare_one_debug_tensor(
                "debug_step0_q_rope_ref_vs_shallow",
                ref_state["q_rope"].unsqueeze(1),          # [1,1,H,rope]
                out_shallow["q_rope"],                     # [1,1,H,rope]
            )
            compare_one_debug_tensor(
                "debug_step0_kv_latent_curtok_ref_vs_shallow",
                ref_state["cache_latent_1tok"].view(1, 1, -1),   # [1,1,kv]
                out_shallow["normalized_kv_latent"],             # [1,1,kv]
            )
            compare_one_debug_tensor(
                "debug_step0_k_rope_curtok_ref_vs_shallow",
                ref_state["cache_k_rope_1tok"].view(1, 1, -1),   # [1,1,rope]
                out_shallow["k_rope"],                           # [1,1,rope]
            )

            ref_scores = (
                torch.einsum(
                    "bhk,tk->bht",
                    ref_state["q_nope_absorb"].float(),
                    cache_latent.float(),
                )
                + torch.einsum(
                    "bhr,tr->bht",
                    ref_state["q_rope"].float(),
                    cache_k_rope.float(),
                )
            ) / ((kv_lora_rank + qk_rope_head_dim) ** 0.5)

            compare_one_debug_tensor(
                "debug_step0_scores_ref_vs_shallow",
                ref_scores.unsqueeze(1),   # [1,1,H,T]
                out_shallow["scores"],     # [1,1,H,T]
            )
            compare_one_debug_tensor(
                "debug_step0_latent_ref_vs_shallow",
                y_ref_latent,
                y_shallow_latent,
            )
            compare_one_debug_tensor(
                "debug_step0_final_ref_vs_shallow",
                y_ref_hidden,
                y_shallow_hidden,
            )

        ys_ref_latent.append(y_ref_latent)
        ys_shallow_latent.append(y_shallow_latent)
        ys_ref_hidden.append(y_ref_hidden)
        ys_shallow_hidden.append(y_shallow_hidden)

        compare_arrays(
            f"decode_step_{i}_eager_absorbed_vs_shallowmla_latent",
            y_ref_latent.detach().float().cpu().numpy(),
            y_shallow_latent.detach().float().cpu().numpy(),
        )
        compare_arrays(
            f"decode_step_{i}_eager_absorbed_vs_shallowmla_final",
            y_ref_hidden.detach().float().cpu().numpy(),
            y_shallow_hidden.detach().float().cpu().numpy(),
        )

    y_ref_latent_all = torch.cat(ys_ref_latent, dim=1)
    y_shallow_latent_all = torch.cat(ys_shallow_latent, dim=1)
    y_ref_hidden_all = torch.cat(ys_ref_hidden, dim=1)
    y_shallow_hidden_all = torch.cat(ys_shallow_hidden, dim=1)

    compare_arrays(
        "decode_all_steps_eager_absorbed_vs_shallowmla_latent",
        y_ref_latent_all.detach().float().cpu().numpy(),
        y_shallow_latent_all.detach().float().cpu().numpy(),
    )
    compare_arrays(
        "decode_all_steps_eager_absorbed_vs_shallowmla_final",
        y_ref_hidden_all.detach().float().cpu().numpy(),
        y_shallow_hidden_all.detach().float().cpu().numpy(),
    )

    print("ALL ABSORBED-LATENT VS SHALLOWMLA CASES PASSED")


if __name__ == "__main__":
    main()
