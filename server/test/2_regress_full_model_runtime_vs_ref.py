from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from server.backbone_store import BackboneLoadPlan, preload_non_moe_backbone
from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.absorbed_latent_ref import run_attention_block_ref
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.full_model_types import ModelExecResult
from server.full_model_runtime import run_full_model
from server.inference_session import InferenceSession
from server.tensor_cache import MappedTensorStore
from server.test.utils import compare_arrays, print_stats, to_numpy_f32


FULL_MODEL_LAYER_COS_MIN = 0.9998
FULL_MODEL_FINAL_COS_MIN = 0.9998
FULL_MODEL_FINAL_MEAN_ABS_MAX = 0.08
FULL_MODEL_FINAL_MAX_ABS_MAX = 1.0


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


def build_layer_map(per_layer):
    out = {}
    for item in per_layer:
        if "layer_id" in item:
            out[int(item["layer_id"])] = to_numpy_f32(item["output"])
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
    hidden = prepared["hidden_in"]

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
        "final_hidden": to_numpy_f32(result["output"]),
        "per_layer": result.get("per_layer", []),
        "logits": to_numpy_f32(logits_result.output),
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
    hidden = prepared["hidden_in"]

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
        "final_hidden": to_numpy_f32(result["output"]),
        "per_layer": result.get("per_layer", []),
        "logits": to_numpy_f32(logits_result.output),
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

    last_layer_metrics = None

    for layer_id in common_layers:
        rt = to_numpy_f32(rt_layers[layer_id]).reshape(-1)
        rf = to_numpy_f32(rf_layers[layer_id]).reshape(-1)
     
        print_stats(f"runtime.layer{layer_id}_output", rt)
        print_stats(f"ref.layer{layer_id}_output", rf)
        compare_arrays(f"layer{layer_id}_output", rf, rt)
     
        diff = np.abs(rf - rt)
        mean_abs = float(diff.mean())
        max_abs = float(diff.max())
        cos = float(
            np.dot(rf, rt) / (np.linalg.norm(rf) * np.linalg.norm(rt) + 1e-12)
        )
     
        assert cos >= FULL_MODEL_LAYER_COS_MIN, (
            f"layer{layer_id} cosine too low: got={cos:.8f} "
            f"expected>={FULL_MODEL_LAYER_COS_MIN:.8f}"
        )
     
        if layer_id == common_layers[-1]:
            last_layer_metrics = {
                "layer_id": layer_id,
                "mean_abs": mean_abs,
                "max_abs": max_abs,
                "cos": cos,
            }

    if last_layer_metrics is None:
        raise RuntimeError("no common per-layer outputs found for regression check")

    assert last_layer_metrics["cos"] >= FULL_MODEL_FINAL_COS_MIN, (
        f"final layer cosine too low: "
        f"layer={last_layer_metrics['layer_id']} "
        f"got={last_layer_metrics['cos']:.8f} "
        f"expected>={FULL_MODEL_FINAL_COS_MIN:.8f}"
    )

    assert last_layer_metrics["mean_abs"] <= FULL_MODEL_FINAL_MEAN_ABS_MAX, (
        f"final layer mean_abs too high: "
        f"layer={last_layer_metrics['layer_id']} "
        f"got={last_layer_metrics['mean_abs']:.6e} "
        f"expected<={FULL_MODEL_FINAL_MEAN_ABS_MAX:.6e}"
    )

    assert last_layer_metrics["max_abs"] <= FULL_MODEL_FINAL_MAX_ABS_MAX, (
        f"final layer max_abs too high: "
        f"layer={last_layer_metrics['layer_id']} "
        f"got={last_layer_metrics['max_abs']:.6e} "
        f"expected<={FULL_MODEL_FINAL_MAX_ABS_MAX:.6e}"
    )

    print(
        "[regress] full_model passed: "
        f"final_layer={last_layer_metrics['layer_id']} "
        f"cos={last_layer_metrics['cos']:.8f} "
        f"mean_abs={last_layer_metrics['mean_abs']:.6e} "
        f"max_abs={last_layer_metrics['max_abs']:.6e}"
    )

    rf_final = to_numpy_f32(ref_out["final_hidden"]).reshape(-1)
    rt_final = to_numpy_f32(runtime_out["final_hidden"]).reshape(-1)
    final_diff = np.abs(rf_final - rt_final)
    final_mean_abs = float(final_diff.mean())
    final_max_abs = float(final_diff.max())
    final_cos = float(
        np.dot(rf_final, rt_final) / (np.linalg.norm(rf_final) * np.linalg.norm(rt_final) + 1e-12)
    )

    assert final_cos >= FULL_MODEL_FINAL_COS_MIN, (
        f"final_hidden cosine too low: got={final_cos:.8f} "
        f"expected>={FULL_MODEL_FINAL_COS_MIN:.8f}"
    )
    assert final_mean_abs <= FULL_MODEL_FINAL_MEAN_ABS_MAX, (
        f"final_hidden mean_abs too high: got={final_mean_abs:.6e} "
        f"expected<={FULL_MODEL_FINAL_MEAN_ABS_MAX:.6e}"
    )
    assert final_max_abs <= FULL_MODEL_FINAL_MAX_ABS_MAX, (
        f"final_hidden max_abs too high: got={final_max_abs:.6e} "
        f"expected<={FULL_MODEL_FINAL_MAX_ABS_MAX:.6e}"
    )


if __name__ == "__main__":
    main()
