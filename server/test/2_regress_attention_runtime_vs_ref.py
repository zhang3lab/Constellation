from __future__ import annotations

import argparse
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
from server.test.utils import compare_arrays, prenorm_hidden_for_attention, print_stats, to_numpy_f32


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


def _align_semantic_shape(a: Any, b: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    a_np = to_numpy_f32(a)
    b_np = to_numpy_f32(b)

    a_sq = np.squeeze(a_np)
    b_sq = np.squeeze(b_np)

    if a_sq.shape != b_sq.shape:
        raise RuntimeError(
            f"{name}: semantic shape mismatch after squeeze: "
            f"{tuple(a_np.shape)} -> {tuple(a_sq.shape)} vs "
            f"{tuple(b_np.shape)} -> {tuple(b_sq.shape)}"
        )

    return a_sq, b_sq


def _compare_aligned(name: str, ref_v: Any, rt_v: Any) -> None:
    ref_aligned, rt_aligned = _align_semantic_shape(ref_v, rt_v, name)

    print_stats(f"ref.{name}", ref_aligned)
    print_stats(f"runtime.{name}", rt_aligned)
    compare_arrays(name, ref_aligned, rt_aligned)


def _extract_prefill_last_hidden(prefill_result: PrefillResult) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(prefill_result, PrefillResult):
        raise TypeError(
            f"prefill_result expected PrefillResult, got {type(prefill_result).__name__}"
        )

    aux = prefill_result.aux
    if aux is None:
        raise RuntimeError("prefill_result.aux must not be None")

    input_ids = aux.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise RuntimeError("prefill_result.aux['input_ids'] must be torch.Tensor")
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise RuntimeError(
                f"prefill_result.aux['input_ids'] expected shape [T] or [1, T], got {tuple(input_ids.shape)}"
            )
    elif input_ids.ndim != 1:
        raise RuntimeError(
            f"prefill_result.aux['input_ids'] expected shape [T] or [1, T], got {tuple(input_ids.shape)}"
        )
    if input_ids.numel() <= 0:
        raise RuntimeError("prefill_result.aux['input_ids'] must be non-empty")

    last_hidden = aux.get("last_hidden")
    if not isinstance(last_hidden, torch.Tensor):
        raise TypeError(
            f"prefill_result.aux['last_hidden'] expected torch.Tensor, got {type(last_hidden).__name__}"
        )
    if last_hidden.ndim != 1:
        raise RuntimeError(
            f"prefill_result.aux['last_hidden'] expected shape [H], got {tuple(last_hidden.shape)}"
        )

    return input_ids, last_hidden


def produce_prefill_last_hidden(
    session: InferenceSession,
    *,
    prompt: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    session.full_model_executor = DeepseekFullModelExecutor(session)

    kv_cache_cfg = session.cfg["kv_cache"]
    session.initialize_full_model_runtime(
        tensor_cache_dir="tmp/non_moe_backbone_cache",
        split_layer=30,
        backbone_dtype=torch.bfloat16,
        kv_cache_cfg=kv_cache_cfg,
    )
    session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

    if session.page_attention_cache_managers is None:
        raise RuntimeError("session.page_attention_cache_managers is not initialized")
    if not isinstance(session.page_attention_cache_managers, dict):
        raise RuntimeError("session.page_attention_cache_managers must be a dict")

    prefill_result = run_prefill(
        session,
        prompt=prompt,
        start_layer=0,
        end_layer=60,
        kv_cache=session.page_attention_cache_managers,
        collect_per_layer=False,
    )
    return _extract_prefill_last_hidden(prefill_result)


def run_runtime_attention(
    session: InferenceSession,
    *,
    hidden_t: torch.Tensor,
    input_ids: torch.Tensor,
    layer_id: int,
    position_id: int,
) -> dict[str, Any]:
    session.full_model_executor = DeepseekFullModelExecutor(session)

    kv_cache_cfg = session.cfg["kv_cache"]
    session.initialize_full_model_runtime(
        tensor_cache_dir="tmp/non_moe_backbone_cache",
        split_layer=30,
        backbone_dtype=torch.bfloat16,
        kv_cache_cfg=kv_cache_cfg,
    )
    session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

    if session.page_attention_cache_managers is None:
        raise RuntimeError("session.page_attention_cache_managers is not initialized")
    if not isinstance(session.page_attention_cache_managers, dict):
        raise RuntimeError("session.page_attention_cache_managers must be a dict")

    hidden_prenorm = prenorm_hidden_for_attention(session, hidden_t, layer_id)
    pos = torch.tensor([int(position_id)], dtype=torch.int64)

    out = session.full_model_executor.run_attention_block(
        hidden_prenorm,
        layer_id,
        position_ids=pos,
        attention_mask=None,
        kv_cache=session.page_attention_cache_managers,
        return_aux=True,
    )

    return {
        "input_ids": input_ids,
        "hidden_in": to_numpy_f32(hidden_t),
        "hidden_prenorm": to_numpy_f32(hidden_prenorm),
        "output": to_numpy_f32(out.output),
        "aux": out.aux or {},
    }


def run_reference_attention(
    session: InferenceSession,
    *,
    hidden_t: torch.Tensor,
    input_ids: torch.Tensor,
    layer_id: int,
    position_id: int,
) -> dict[str, Any]:
    session.full_model_executor = DeepseekFullModelReference(session)

    kv_cache_cfg = session.cfg["kv_cache"]

    session.initialize_full_model_runtime(
        tensor_cache_dir="tmp/non_moe_backbone_cache",
        split_layer=30,
        backbone_dtype=torch.float32,
        kv_cache_cfg=kv_cache_cfg,
        plan=BackboneLoadPlan.attention_only(
            attention_dtype=torch.float32,
            embed_dtype=torch.float32,
            layer_ids={int(layer_id)},
        ),
    )

    if session.page_attention_cache_managers is None:
        raise RuntimeError("session.page_attention_cache_managers is not initialized")
    if not isinstance(session.page_attention_cache_managers, dict):
        raise RuntimeError("session.page_attention_cache_managers must be a dict")

    hidden_prenorm = prenorm_hidden_for_attention(session, hidden_t, layer_id)
    pos = torch.tensor([int(position_id)], dtype=torch.int64)

    out = session.full_model_executor.run_attention_block(
        hidden_prenorm,
        layer_id,
        position_ids=pos,
        attention_mask=None,
        kv_cache=session.page_attention_cache_managers,
        return_aux=True,
    )

    return {
        "input_ids": input_ids,
        "hidden_in": to_numpy_f32(hidden_t),
        "hidden_prenorm": to_numpy_f32(hidden_prenorm),
        "output": to_numpy_f32(out.output),
        "aux": out.aux or {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--layer-id", type=int, default=3)
    ap.add_argument("--position-id", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    with InferenceSession(coord, cfg) as prefill_sess:
        input_ids, hidden_t = produce_prefill_last_hidden(
            prefill_sess,
            prompt=args.prompt,
        )

    with InferenceSession(coord, cfg) as runtime_sess:
        runtime_out = run_runtime_attention(
            runtime_sess,
            hidden_t=hidden_t,
            input_ids=input_ids,
            layer_id=args.layer_id,
            position_id=args.position_id,
        )

    with InferenceSession(coord, cfg) as ref_sess:
        ref_out = run_reference_attention(
            ref_sess,
            hidden_t=hidden_t,
            input_ids=input_ids,
            layer_id=args.layer_id,
            position_id=args.position_id,
        )

    print(f"[compare] prompt={args.prompt!r}")
    print(f"[compare] input_ids={input_ids.reshape(-1).tolist()}")

    _compare_aligned("hidden_in", ref_out["hidden_in"], runtime_out["hidden_in"])
    _compare_aligned("hidden_prenorm", ref_out["hidden_prenorm"], runtime_out["hidden_prenorm"])
    _compare_aligned("attention_output", ref_out["output"], runtime_out["output"])

    ref_aux = ref_out["aux"]
    rt_aux = runtime_out["aux"]

    if "q_flash" in ref_aux and "q_flash" in rt_aux:
        _compare_aligned("attention.q_flash", ref_aux["q_flash"], rt_aux["q_flash"])
    else:
        print("[skip] q_flash missing in ref/runtime aux")

    if "blocked_k_token" in ref_aux and "blocked_k_token" in rt_aux:
        _compare_aligned(
            "attention.blocked_k_token",
            ref_aux["blocked_k_token"],
            rt_aux["blocked_k_token"],
        )
    else:
        print("[skip] blocked_k_token missing in ref/runtime aux")


if __name__ == "__main__":
    main()
