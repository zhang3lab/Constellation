"""
Unified compare entry for absorbed-latent reference vs attention backends.

Supports:
- shallowmla
- flashmla
- all
"""
from __future__ import annotations

import argparse
import math
from typing import Optional

import numpy as np
import torch

from server.absorbed_latent_ref import (
    build_ref_state_for_one_token,
    eager_absorbed_latent_attention,
    latent_to_final_hidden,
)
from server.deepseek_model_loader import DeepseekModelLoader
from server.test.utils import compare_arrays

from third_party.ShallowMLA.mla import MLA, precompute_freqs_cis


class ShallowMLAAttentionWrapper:
    def __init__(
        self,
        *,
        model_loader,
        layer_id: int,
        max_batch_size: int = 4,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        optim_type: str = "torch",
    ):
        self.layer_id = int(layer_id)
        self.device = torch.device(device)
        self.dtype = dtype
        self.optim_type = str(optim_type)

        mla_cfg = model_loader.mla_config()

        self.mla = MLA(
            dim=mla_cfg["dim"],
            kv_latent_rank=mla_cfg["kv_latent_rank"],
            q_latent_rank=mla_cfg["q_latent_rank"],
            num_heads=mla_cfg["num_heads"],
            qk_nrope_head_dim=mla_cfg["qk_nrope_head_dim"],
            v_head_dim=mla_cfg["v_head_dim"],
            qk_rope_head_dim=mla_cfg["qk_rope_head_dim"],
            max_batch_size=max_batch_size,
            max_seq_len=mla_cfg["max_seq_len"],
            dtype=dtype,
            optim_type=self.optim_type,
            eps=1e-6,
            use_page_cache=False,
            use_page_cache_triton=False,
        ).to(self.device)

        ws = model_loader.load_attention_block_weights_fp32(self.layer_id)

        with torch.no_grad():
            self.mla.proj_q_down.weight.copy_(ws["q_a_proj"].to(device=self.device, dtype=dtype))
            self.mla.rms_norm_q_weight.copy_(ws["q_a_layernorm"].to(device=self.device, dtype=dtype))
            self.mla.proj_q_up.weight.copy_(ws["q_b_proj"].to(device=self.device, dtype=dtype))

            self.mla.proj_kv_down.weight.copy_(ws["kv_a_proj_with_mqa"].to(device=self.device, dtype=dtype))
            self.mla.rms_norm_kv_weight.copy_(ws["kv_a_layernorm"].to(device=self.device, dtype=dtype))
            self.mla.proj_kv_up.weight.copy_(ws["kv_b_proj"].to(device=self.device, dtype=dtype))

            self.mla.proj_out.weight.copy_(ws["o_proj"].to(device=self.device, dtype=dtype))

        self.input_layernorm_weight = ws["input_layernorm"].to(device=self.device, dtype=dtype)
        self.eps = 1e-6

        self.freq_cis = precompute_freqs_cis(
            qk_rope_head_dim=mla_cfg["qk_rope_head_dim"],
            seq_len=mla_cfg["max_seq_len"],
            seq_len_train=mla_cfg["max_seq_len_train"],
            beta_fast=mla_cfg["beta_fast"],
            beta_slow=mla_cfg["beta_slow"],
            rope_theta=mla_cfg["rope_theta"],
            rope_factor=mla_cfg["rope_factor"],
            mscale=mla_cfg.get("mscale", 1.0),
            mscale_all_dim=mla_cfg.get("mscale_all_dim", 1.0),
            dtype=dtype,
        ).to(self.device)

    def rms_norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(
            x,
            (x.shape[-1],),
            self.input_layernorm_weight,
            self.eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.rms_norm_input(x)
        seq_len = int(x.shape[1])
        freq_cis = self.freq_cis[start_pos:start_pos + seq_len]
        return self.mla(
            x,
            start_pos=start_pos,
            freq_cis=freq_cis,
            mask=mask,
        )

    def forward_debug(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ):
        x = self.rms_norm_input(x)
        seq_len = int(x.shape[1])
        freq_cis = self.freq_cis[start_pos:start_pos + seq_len]
        return self.mla(
            x,
            start_pos=start_pos,
            freq_cis=freq_cis,
            mask=mask,
            return_debug=True,
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


class RefFlashCache:
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


def compare_shallowmla(args):
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
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        ).unsqueeze(1)

        y_ref_hidden = latent_to_final_hidden(
            y_ref_latent[:, 0],
            ws,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
        ).unsqueeze(1)

        out_shallow = shallow.forward_debug(x_step, start_pos=pos)
        y_shallow_latent = out_shallow["latent"]
        y_shallow_hidden = out_shallow["hidden"]

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


def compare_flashmla(args):
    import flash_mla

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
    d = kv_lora_rank + qk_rope_head_dim
    dv = kv_lora_rank
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
        ).unsqueeze(1)
        ys_ref.append(y_ref)

        q_flash = ref_state["q_flash"].unsqueeze(1)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=str, required=True)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--backend", type=str, default="all", choices=["shallowmla", "flashmla", "all"])
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optim-type", type=str, default="triton", choices=["torch", "triton"])
    args = parser.parse_args()

    if args.backend in ("shallowmla", "all"):
        compare_shallowmla(args)

    if args.backend in ("flashmla", "all"):
        if args.dtype == "float32":
            raise RuntimeError("flashmla backend only supports bfloat16/float16")
        compare_flashmla(args)


if __name__ == "__main__":
    main()
