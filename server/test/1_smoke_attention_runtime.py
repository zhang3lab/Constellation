from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from server.absorbed_latent_ref import run_attention_block_ref
from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.full_model_types import ModelExecResult
from server.inference_session import InferenceSession
from server.test.utils import compare_arrays, print_stats


class DeepseekFullModelReference(DeepseekFullModelExecutor):
    def run_attention_block(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        position_ids=None,
        attention_mask=None,
        kv_cache=None,
        return_aux: bool = False,
    ) -> ModelExecResult:
        return run_attention_block_ref(
            self.session,
            hidden_in,
            int(layer_id),
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            return_aux=return_aux,
        )


def run_runtime_attention(
    session: InferenceSession,
    *,
    prompt: str,
    layer_id: int,
    position_id: int,
) -> dict[str, Any]:
    session.full_model_executor = DeepseekFullModelExecutor(session)

    kv_cache_cfg = session.cfg["kv_cache"]
    session.ensure_full_model_runtime(
        tensor_cache_dir="tmp/non_moe_backbone_cache",
        split_layer=30,
        backbone_dtype=torch.bfloat16,
        kv_cache_cfg=kv_cache_cfg,
    )
    session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

    prepared = session.full_model_executor.prepare_prompt_hidden_input(prompt)
    hidden = np.asarray(prepared["hidden_in"], dtype=np.float32)

    pos = np.asarray([position_id], dtype=np.int64)

    out = session.full_model_executor.run_attention_block(
        hidden,
        layer_id,
        position_ids=pos,
        attention_mask=None,
        kv_cache=session.page_attention_cache_managers,
        return_aux=True,
    )

    return {
        "hidden_in": hidden,
        "output": np.asarray(out.output, dtype=np.float32),
        "aux": out.aux,
    }


def run_reference_attention(
    session: InferenceSession,
    *,
    prompt: str,
    layer_id: int,
    position_id: int,
) -> dict[str, Any]:
    session.full_model_executor = DeepseekFullModelReference(session)

    kv_cache_cfg = session.cfg["kv_cache"]
    session.ensure_full_model_runtime(
        tensor_cache_dir="tmp/non_moe_backbone_cache",
        split_layer=30,
        backbone_dtype=torch.bfloat16,
        kv_cache_cfg=kv_cache_cfg,
    )
    session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

    prepared = session.full_model_executor.prepare_prompt_hidden_input(prompt)
    hidden = np.asarray(prepared["hidden_in"], dtype=np.float32)

    pos = np.asarray([position_id], dtype=np.int64)

    out = session.full_model_executor.run_attention_block(
        hidden,
        layer_id,
        position_ids=pos,
        attention_mask=None,
        kv_cache=session.page_attention_cache_managers,
        return_aux=True,
    )

    return {
        "hidden_in": hidden,
        "output": np.asarray(out.output, dtype=np.float32),
        "aux": out.aux,
    }


def maybe_compare_aux_tensor(name: str, ref_aux: dict[str, Any], rt_aux: dict[str, Any], key: str) -> None:
    if key not in ref_aux:
        print(f"[skip] ref aux missing {key}")
        return
    if key not in rt_aux:
        print(f"[skip] runtime aux missing {key}")
        return

    ref_v = ref_aux[key]
    rt_v = rt_aux[key]

    print_stats(f"ref.{name}.{key}", ref_v)
    print_stats(f"runtime.{name}.{key}", rt_v)
    compare_arrays(f"{name}.{key}", ref_v, rt_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--layer-id", type=int, default=3)
    ap.add_argument("--position-id", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    with InferenceSession(coord, cfg) as runtime_sess:
        runtime_out = run_runtime_attention(
            runtime_sess,
            prompt=args.prompt,
            layer_id=args.layer_id,
            position_id=args.position_id,
        )

    with InferenceSession(coord, cfg) as ref_sess:
        ref_out = run_reference_attention(
            ref_sess,
            prompt=args.prompt,
            layer_id=args.layer_id,
            position_id=args.position_id,
        )

    print_stats("runtime.hidden_in", runtime_out["hidden_in"])
    print_stats("ref.hidden_in", ref_out["hidden_in"])
    compare_arrays("hidden_in", ref_out["hidden_in"], runtime_out["hidden_in"])

    print_stats("runtime.attention_output", runtime_out["output"])
    print_stats("ref.attention_output", ref_out["output"])
    compare_arrays("attention_output", ref_out["output"], runtime_out["output"])

    ref_aux = ref_out["aux"] or {}
    rt_aux = runtime_out["aux"] or {}

    maybe_compare_aux_tensor("attention", ref_aux, rt_aux, "q_flash")
    maybe_compare_aux_tensor("attention", ref_aux, rt_aux, "blocked_k_token")


if __name__ == "__main__":
    main()
