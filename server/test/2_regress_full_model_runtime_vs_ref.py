from __future__ import annotations

import argparse
import gc
from typing import Any

import numpy as np
import torch

from server.absorbed_latent_ref import run_attention_block_ref
from server.backbone_store import BackboneLoadPlan
from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.full_model_types import ModelExecResult
from server.generation_types import PrefillResult
from server.inference_session import InferenceSession
from server.prefill_runtime import run_prefill
from server.test.utils import compare_arrays, print_stats, to_numpy_f32


FULL_MODEL_LAYER_COS_MIN = 0.99975
FULL_MODEL_FINAL_COS_MIN = 0.99975
FULL_MODEL_FINAL_MEAN_ABS_MAX = 0.08
FULL_MODEL_FINAL_MAX_ABS_MAX = 1.0


class DeepseekFullModelReference(DeepseekFullModelExecutor):
    def run_attention_block(
        self,
        hidden_in,
        layer_id: int,
        *,
        position_ids=None,
        attention_mask=None,
        kv_cache=None,
        return_aux: bool = False,
    ) -> ModelExecResult:
        if isinstance(hidden_in, np.ndarray):
            hidden_t = torch.from_numpy(hidden_in)
        else:
            hidden_t = hidden_in

        if not isinstance(hidden_t, torch.Tensor):
            raise TypeError(
                f"hidden_in expected torch.Tensor or np.ndarray, got {type(hidden_in).__name__}"
            )

        if hidden_t.ndim == 1:
            return run_attention_block_ref(
                self.session,
                hidden_t,
                int(layer_id),
                position_ids=position_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                return_aux=return_aux,
            )

        if hidden_t.ndim != 2:
            raise RuntimeError(
                f"run_attention_block hidden_in must be 1D or 2D, got shape={tuple(hidden_t.shape)}"
            )

        seq_len = int(hidden_t.shape[0])
        if seq_len <= 0:
            raise RuntimeError("run_attention_block hidden_in must have positive seq_len")

        if position_ids is None:
            pos_list = list(range(seq_len))
        elif isinstance(position_ids, torch.Tensor):
            pos_list = [int(x) for x in position_ids.reshape(-1).tolist()]
        else:
            pos_list = [int(x) for x in np.asarray(position_ids).reshape(-1).tolist()]

        if len(pos_list) != seq_len:
            raise RuntimeError(
                f"position_ids length mismatch: len(position_ids)={len(pos_list)} seq_len={seq_len}"
            )

        outputs = []
        aux_steps = []

        for i in range(seq_len):
            step_hidden = hidden_t[i]
            step_pos = np.asarray([pos_list[i]], dtype=np.int64)

            step_result = run_attention_block_ref(
                self.session,
                step_hidden,
                int(layer_id),
                position_ids=step_pos,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                return_aux=return_aux,
            )

            step_out = step_result.output
            if isinstance(step_out, np.ndarray):
                step_out = torch.from_numpy(step_out)
            if not isinstance(step_out, torch.Tensor):
                raise TypeError(
                    f"step output expected torch.Tensor or np.ndarray, got {type(step_result.output).__name__}"
                )
            if step_out.ndim != 1:
                raise RuntimeError(
                    f"step output must be 1D, got shape={tuple(step_out.shape)}"
                )

            outputs.append(step_out)
            if return_aux:
                aux_steps.append(step_result.aux or {})

        stacked = torch.stack(outputs, dim=0)

        aux = {}
        if return_aux:
            aux = {
                "kind": "attention_block_ref",
                "layer_id": int(layer_id),
                "seq_len": seq_len,
                "positions": pos_list,
                "steps": aux_steps,
            }
            if aux_steps:
                last_aux = aux_steps[-1]
                for k, v in last_aux.items():
                    aux.setdefault(k, v)

        return ModelExecResult(output=stacked, aux=aux)


def build_layer_map(per_layer):
    out = {}
    if per_layer is None:
        return out

    if isinstance(per_layer, dict):
        for key, value in per_layer.items():
            layer_id = int(str(key).split("_")[-1]) if isinstance(key, str) else int(key)
            if isinstance(value, dict) and "hidden_out" in value:
                out[layer_id] = to_numpy_f32(value["hidden_out"])
        return out

    for item in per_layer:
        if "layer_id" in item:
            out[int(item["layer_id"])] = to_numpy_f32(item["output"])
    return out


def _require_prefill_result(x: Any, name: str) -> PrefillResult:
    if not isinstance(x, PrefillResult):
        raise TypeError(f"{name} expected PrefillResult, got {type(x).__name__}")
    return x


def _extract_prefill_outputs(prefill_result: PrefillResult) -> dict[str, Any]:
    aux = prefill_result.aux
    if aux is None:
        raise RuntimeError("prefill_result.aux must not be None")

    input_ids = aux.get("input_ids")
    if not isinstance(input_ids, list) or not input_ids:
        raise RuntimeError("prefill_result.aux['input_ids'] must be a non-empty list[int]")
    if not all(isinstance(x, int) for x in input_ids):
        raise RuntimeError("prefill_result.aux['input_ids'] must be list[int]")

    final_hidden_t = aux.get("final_hidden")
    last_hidden_t = aux.get("last_hidden")
    logits_t = prefill_result.next_token_logits

    if not isinstance(final_hidden_t, torch.Tensor):
        raise TypeError(
            f"prefill_result.aux['final_hidden'] expected torch.Tensor, got {type(final_hidden_t).__name__}"
        )
    if not isinstance(last_hidden_t, torch.Tensor):
        raise TypeError(
            f"prefill_result.aux['last_hidden'] expected torch.Tensor, got {type(last_hidden_t).__name__}"
        )
    if not isinstance(logits_t, torch.Tensor):
        raise TypeError(
            f"prefill_result.next_token_logits expected torch.Tensor, got {type(logits_t).__name__}"
        )

    if final_hidden_t.ndim != 2:
        raise RuntimeError(
            f"prefill_result.aux['final_hidden'] expected shape [T, H], got {tuple(final_hidden_t.shape)}"
        )
    if last_hidden_t.ndim != 1:
        raise RuntimeError(
            f"prefill_result.aux['last_hidden'] expected shape [H], got {tuple(last_hidden_t.shape)}"
        )
    if logits_t.ndim != 1:
        raise RuntimeError(
            f"prefill_result.next_token_logits expected shape [V], got {tuple(logits_t.shape)}"
        )

    if final_hidden_t.shape[0] != len(input_ids):
        raise RuntimeError(
            f"final_hidden length mismatch: T={final_hidden_t.shape[0]} len(input_ids)={len(input_ids)}"
        )

    return {
        "input_ids": [int(x) for x in input_ids],
        "final_hidden": to_numpy_f32(final_hidden_t),
        "last_hidden": to_numpy_f32(last_hidden_t),
        "per_layer": aux.get("per_layer"),
        "logits": to_numpy_f32(logits_t),
    }


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
    session.initialize_full_model_runtime(
        tensor_cache_dir="tmp/non_moe_backbone_cache",
        split_layer=30,
        backbone_dtype=torch.bfloat16,
        kv_cache_cfg=kv_cache_cfg,
    )

    if session.page_attention_cache_managers is None:
        raise RuntimeError("session.page_attention_cache_managers is not initialized")
    if not isinstance(session.page_attention_cache_managers, dict):
        raise RuntimeError("session.page_attention_cache_managers must be a dict")

    prefill_result = _require_prefill_result(
        run_prefill(
            session,
            prompt=prompt,
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache=session.page_attention_cache_managers,
            collect_per_layer=collect_per_layer,
        ),
        "run_prefill(...)",
    )

    return _extract_prefill_outputs(prefill_result)


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

    session.initialize_full_model_runtime(
        tensor_cache_dir="tmp/non_moe_backbone_cache",
        split_layer=30,
        backbone_dtype=torch.float32,
        kv_cache_cfg=kv_cache_cfg,
        plan=BackboneLoadPlan.runtime_fp32_no_attention_no_routed_experts(),
    )

    if session.page_attention_cache_managers is None:
        raise RuntimeError("session.page_attention_cache_managers is not initialized")
    if not isinstance(session.page_attention_cache_managers, dict):
        raise RuntimeError("session.page_attention_cache_managers must be a dict")

    prefill_result = _require_prefill_result(
        run_prefill(
            session,
            prompt=prompt,
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache=session.page_attention_cache_managers,
            collect_per_layer=collect_per_layer,
        ),
        "run_prefill(...)",
    )

    return _extract_prefill_outputs(prefill_result)


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

    gc.collect()
    torch.cuda.empty_cache()

    with InferenceSession(coord, cfg) as ref_sess:
        ref_out = run_reference_path(
            ref_sess,
            prompt=args.prompt,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            collect_per_layer=True,
        )

    print_stats("runtime.final_hidden", runtime_out["final_hidden"])
    print_stats("ref.final_hidden", ref_out["final_hidden"])
    compare_arrays("final_hidden", ref_out["final_hidden"], runtime_out["final_hidden"])

    print_stats("runtime.last_hidden", runtime_out["last_hidden"])
    print_stats("ref.last_hidden", ref_out["last_hidden"])
    compare_arrays("last_hidden", ref_out["last_hidden"], runtime_out["last_hidden"])

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
