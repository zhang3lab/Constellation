from __future__ import annotations

import argparse
import gc

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
FULL_MODEL_FINAL_MAX_ABS_MAX = 3.0


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
        x = torch.from_numpy(hidden_in) if isinstance(hidden_in, np.ndarray) else hidden_in
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"hidden_in expected torch.Tensor or np.ndarray, got {type(hidden_in).__name__}")

        if x.ndim == 1:
            return run_attention_block_ref(
                self.session,
                x,
                int(layer_id),
                position_ids=position_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                return_aux=return_aux,
            )

        if x.ndim != 2:
            raise RuntimeError(f"run_attention_block hidden_in must be 1D or 2D, got shape={tuple(x.shape)}")

        pos = list(range(int(x.shape[0]))) if position_ids is None else [
            int(v) for v in (
                position_ids.reshape(-1).tolist()
                if isinstance(position_ids, torch.Tensor)
                else np.asarray(position_ids).reshape(-1).tolist()
            )
        ]
        if len(pos) != int(x.shape[0]):
            raise RuntimeError(f"position_ids length mismatch: len(position_ids)={len(pos)} seq_len={int(x.shape[0])}")

        ys = []
        aux_steps = []
        for i in range(int(x.shape[0])):
            r = run_attention_block_ref(
                self.session,
                x[i],
                int(layer_id),
                position_ids=np.asarray([pos[i]], dtype=np.int64),
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                return_aux=return_aux,
            )
            y = torch.from_numpy(r.output) if isinstance(r.output, np.ndarray) else r.output
            ys.append(y)
            if return_aux:
                aux_steps.append(r.aux or {})

        aux = {}
        if return_aux:
            aux = {"kind": "attention_block_ref", "layer_id": int(layer_id), "seq_len": int(x.shape[0]), "positions": pos, "steps": aux_steps}
            if aux_steps:
                aux.update({k: v for k, v in aux_steps[-1].items() if k not in aux})

        return ModelExecResult(output=torch.stack(ys, dim=0), aux=aux)


def build_layer_map(per_layer):
    out = {}
    if per_layer is None:
        return out
    if isinstance(per_layer, dict):
        for k, v in per_layer.items():
            lid = int(str(k).split("_")[-1]) if isinstance(k, str) else int(k)
            if isinstance(v, dict) and "hidden_out" in v:
                out[lid] = to_numpy_f32(v["hidden_out"])
        return out
    for item in per_layer:
        if "layer_id" in item:
            out[int(item["layer_id"])] = to_numpy_f32(item["output"])
    return out


def _extract_prefill_outputs(prefill_result: PrefillResult):
    if not isinstance(prefill_result, PrefillResult):
        raise TypeError(f"prefill_result expected PrefillResult, got {type(prefill_result).__name__}")
    aux = prefill_result.aux or {}
    input_ids = aux.get("input_ids")
    final_hidden = aux.get("final_hidden")
    last_hidden = aux.get("last_hidden")
    logits = prefill_result.next_token_logits

    if not isinstance(input_ids, torch.Tensor):
        raise RuntimeError("prefill_result.aux['input_ids'] must be torch.Tensor")
    if input_ids.ndim == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids[0]
    if input_ids.ndim != 1 or input_ids.numel() <= 0:
        raise RuntimeError(f"prefill_result.aux['input_ids'] expected shape [T], got {tuple(input_ids.shape)}")
    if not isinstance(final_hidden, torch.Tensor) or final_hidden.ndim != 2:
        raise RuntimeError("prefill_result.aux['final_hidden'] must be [T, H] torch.Tensor")
    if not isinstance(last_hidden, torch.Tensor) or last_hidden.ndim != 1:
        raise RuntimeError("prefill_result.aux['last_hidden'] must be [H] torch.Tensor")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise RuntimeError("prefill_result.next_token_logits must be [V] torch.Tensor")

    return {
        "input_ids": input_ids,
        "final_hidden": to_numpy_f32(final_hidden),
        "last_hidden": to_numpy_f32(last_hidden),
        "per_layer": aux.get("per_layer"),
        "logits": to_numpy_f32(logits),
    }


def _run_path(session: InferenceSession, *, prompt: str, start_layer: int, end_layer: int, collect_per_layer: bool, ref: bool):
    session.full_model_executor = DeepseekFullModelReference(session) if ref else DeepseekFullModelExecutor(session)
    kv_cache_cfg = session.cfg["kv_cache"]
    kwargs = dict(
        tensor_cache_dir="tmp/non_moe_backbone_cache",
        split_layer=30,
        backbone_dtype=torch.float32 if ref else torch.bfloat16,
        kv_cache_cfg=kv_cache_cfg,
    )
    if ref:
        kwargs["plan"] = BackboneLoadPlan.runtime_fp32_no_attention_no_routed_experts()
    session.initialize_full_model_runtime(**kwargs)

    kv_cache = session.page_attention_cache_managers
    if not isinstance(kv_cache, dict):
        raise RuntimeError("session.page_attention_cache_managers must be a dict")

    return _extract_prefill_outputs(
        run_prefill(
            session,
            prompt=prompt,
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache=kv_cache,
            collect_per_layer=collect_per_layer,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=60)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"], log_level=cfg["log_level"])
    setup_control_plane(coord, cfg)

    with InferenceSession(coord, cfg) as runtime_sess:
        runtime_out = _run_path(
            runtime_sess,
            prompt=args.prompt,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            collect_per_layer=True,
            ref=False,
        )

    gc.collect()
    torch.cuda.empty_cache()

    with InferenceSession(coord, cfg) as ref_sess:
        ref_out = _run_path(
            ref_sess,
            prompt=args.prompt,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            collect_per_layer=True,
            ref=True,
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
    common_layers = sorted(set(rt_layers) & set(rf_layers))
    print(f"[compare] common_layers={common_layers}")

    last = None
    for layer_id in common_layers:
        rt = to_numpy_f32(rt_layers[layer_id]).reshape(-1)
        rf = to_numpy_f32(rf_layers[layer_id]).reshape(-1)
        print_stats(f"runtime.layer{layer_id}_output", rt)
        print_stats(f"ref.layer{layer_id}_output", rf)
        compare_arrays(f"layer{layer_id}_output", rf, rt)

        diff = np.abs(rf - rt)
        cos = float(np.dot(rf, rt) / (np.linalg.norm(rf) * np.linalg.norm(rt) + 1e-12))
        assert cos >= FULL_MODEL_LAYER_COS_MIN, (
            f"layer{layer_id} cosine too low: got={cos:.8f} expected>={FULL_MODEL_LAYER_COS_MIN:.8f}"
        )
        if layer_id == common_layers[-1]:
            last = {
                "layer_id": layer_id,
                "mean_abs": float(diff.mean()),
                "max_abs": float(diff.max()),
                "cos": cos,
            }

    if last is None:
        raise RuntimeError("no common per-layer outputs found for regression check")

    assert last["cos"] >= FULL_MODEL_FINAL_COS_MIN, (
        f"final layer cosine too low: layer={last['layer_id']} got={last['cos']:.8f} expected>={FULL_MODEL_FINAL_COS_MIN:.8f}"
    )
    assert last["mean_abs"] <= FULL_MODEL_FINAL_MEAN_ABS_MAX, (
        f"final layer mean_abs too high: layer={last['layer_id']} got={last['mean_abs']:.6e} expected<={FULL_MODEL_FINAL_MEAN_ABS_MAX:.6e}"
    )
    assert last["max_abs"] <= FULL_MODEL_FINAL_MAX_ABS_MAX, (
        f"final layer max_abs too high: layer={last['layer_id']} got={last['max_abs']:.6e} expected<={FULL_MODEL_FINAL_MAX_ABS_MAX:.6e}"
    )

    print(
        "[regress] full_model passed: "
        f"final_layer={last['layer_id']} cos={last['cos']:.8f} "
        f"mean_abs={last['mean_abs']:.6e} max_abs={last['max_abs']:.6e}"
    )

    rf_final = ref_out["final_hidden"].reshape(-1)
    rt_final = runtime_out["final_hidden"].reshape(-1)
    final_diff = np.abs(rf_final - rt_final)
    final_mean_abs = float(final_diff.mean())
    final_max_abs = float(final_diff.max())
    final_cos = float(np.dot(rf_final, rt_final) / (np.linalg.norm(rf_final) * np.linalg.norm(rt_final) + 1e-12))

    print(
        "[regress] final hidden metrics: "
        f"cos={final_cos:.8f} mean_abs={final_mean_abs:.6e} max_abs={final_max_abs:.6e}"
    )

    assert final_cos >= FULL_MODEL_FINAL_COS_MIN, (
        f"final_hidden cosine too low: got={final_cos:.8f} expected>={FULL_MODEL_FINAL_COS_MIN:.8f}"
    )
    assert final_mean_abs <= FULL_MODEL_FINAL_MEAN_ABS_MAX, (
        f"final_hidden mean_abs too high: got={final_mean_abs:.6e} expected<={FULL_MODEL_FINAL_MEAN_ABS_MAX:.6e}"
    )
    assert final_max_abs <= FULL_MODEL_FINAL_MAX_ABS_MAX, (
        f"final_hidden max_abs too high: got={final_max_abs:.6e} expected<={FULL_MODEL_FINAL_MAX_ABS_MAX:.6e}"
    )


if __name__ == "__main__":
    main()
