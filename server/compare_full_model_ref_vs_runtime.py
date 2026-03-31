from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.absorbed_latent_ref import run_attention_block_ref
from server.deepseek_full_model_ref import DeepseekFullModelExecutor
from server.full_model_ref import ModelExecResult
from server.full_model_runtime import run_full_model
from server.inference_session import InferenceSession
from server.test_utils import compare_arrays, print_stats


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

def build_layer_map(per_layer: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for item in per_layer:
        if "layer_id" in item and "output" in item:
            out[int(item["layer_id"])] = np.asarray(item["output"], dtype=np.float32)
    return out


def run_runtime_path(
    session: InferenceSession,
    *,
    prompt: str,
    start_layer: int,
    end_layer: int,
    collect_per_layer: bool,
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

    result = run_full_model(
        session,
        hidden,
        start_layer=start_layer,
        end_layer=end_layer,
        position_ids=prepared.get("position_ids"),
        attention_mask=prepared.get("attention_mask"),
        kv_cache=session.page_attention_cache_managers,
        collect_per_layer=collect_per_layer,
    )

    logits_result = session.full_model_executor.run_final_norm_and_lm_head(
        result["output"],
        return_aux=False,
    )

    return {
        "prepared": prepared,
        "hidden_in": hidden,
        "final_hidden": np.asarray(result["output"], dtype=np.float32),
        "per_layer": result.get("per_layer", []),
        "logits": np.asarray(logits_result.output, dtype=np.float32),
    }


def run_reference_path(
    session: InferenceSession,
    *,
    prompt: str,
    start_layer: int,
    end_layer: int,
    collect_per_layer: bool,
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

    result = run_full_model(
        session,
        hidden,
        start_layer=start_layer,
        end_layer=end_layer,
        position_ids=prepared.get("position_ids"),
        attention_mask=prepared.get("attention_mask"),
        kv_cache=session.page_attention_cache_managers,
        collect_per_layer=collect_per_layer,
    )

    logits_result = session.full_model_executor.run_final_norm_and_lm_head(
        result["output"],
        return_aux=False,
    )

    return {
        "prepared": prepared,
        "hidden_in": hidden,
        "final_hidden": np.asarray(result["output"], dtype=np.float32),
        "per_layer": result.get("per_layer", []),
        "logits": np.asarray(logits_result.output, dtype=np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=60)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    with InferenceSession(coord, cfg) as runtime_sess:
        runtime_out = run_runtime_path(
            runtime_sess,
            prompt=args.prompt,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            collect_per_layer=True,
        )

    with InferenceSession(coord, cfg) as ref_sess:
        ref_out = run_reference_path(
            ref_sess,
            prompt=args.prompt,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            collect_per_layer=True,
        )

    print_stats("runtime.hidden_in", runtime_out["hidden_in"])
    print_stats("ref.hidden_in", ref_out["hidden_in"])
    compare_arrays("hidden_in", ref_out["hidden_in"], runtime_out["hidden_in"])

    print_stats("runtime.final_hidden", runtime_out["final_hidden"])
    print_stats("ref.final_hidden", ref_out["final_hidden"])
    compare_arrays("final_hidden", ref_out["final_hidden"], runtime_out["final_hidden"])

    print_stats("runtime.logits", runtime_out["logits"])
    print_stats("ref.logits", ref_out["logits"])
    compare_arrays("logits", ref_out["logits"], runtime_out["logits"])

    rt_layers = build_layer_map(runtime_out["per_layer"])
    rf_layers = build_layer_map(ref_out["per_layer"])

    common_layers = sorted(set(rt_layers.keys()) & set(rf_layers.keys()))
    print(f"[compare] common_layers={common_layers}")

    for layer_id in common_layers:
        print_stats(f"runtime.layer{layer_id}_output", rt_layers[layer_id])
        print_stats(f"ref.layer{layer_id}_output", rf_layers[layer_id])
        compare_arrays(
            f"layer{layer_id}_output",
            rf_layers[layer_id],
            rt_layers[layer_id],
        )


if __name__ == "__main__":
    main()
