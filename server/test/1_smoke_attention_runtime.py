from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.generation_types import PrefillResult
from server.inference_session import InferenceSession
from server.prefill_runtime import run_prefill
from server.test.utils import prenorm_hidden_for_attention, print_stats, to_numpy_f32


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

    if not isinstance(prefill_result, PrefillResult):
        raise TypeError(
            f"run_prefill(...) expected PrefillResult, got {type(prefill_result).__name__}"
        )

    aux = prefill_result.aux
    if aux is None:
        raise RuntimeError("prefill_result.aux must not be None")

    input_ids = aux.get("input_ids")
    if not isinstance(input_ids, list) or not input_ids:
        raise RuntimeError("prefill_result.aux['input_ids'] must be a non-empty list[int]")
    if not all(isinstance(x, int) for x in input_ids):
        raise RuntimeError("prefill_result.aux['input_ids'] must be list[int]")

    hidden_t = aux.get("last_hidden")
    if not isinstance(hidden_t, torch.Tensor):
        raise TypeError(
            f"prefill_result.aux['last_hidden'] expected torch.Tensor, got {type(hidden_t).__name__}"
        )
    if hidden_t.ndim != 1:
        raise RuntimeError(
            f"prefill_result.aux['last_hidden'] expected shape [H], got {tuple(hidden_t.shape)}"
        )

    hidden = to_numpy_f32(hidden_t)
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
        "hidden_in": hidden,
        "hidden_prenorm": to_numpy_f32(hidden_prenorm),
        "output": to_numpy_f32(out.output),
        "aux": out.aux or {},
    }


def main() -> None:
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

    input_ids = runtime_out["input_ids"]
    hidden_in = runtime_out["hidden_in"]
    hidden_prenorm = runtime_out["hidden_prenorm"]
    attention_output = runtime_out["output"]
    aux = runtime_out["aux"]

    print(f"[smoke] prompt={args.prompt!r}")
    print(f"[smoke] input_ids={input_ids}")

    print_stats("runtime.hidden_in", hidden_in)
    print_stats("runtime.hidden_prenorm", hidden_prenorm)
    print_stats("runtime.attention_output", attention_output)

    if hidden_in.ndim != 1:
        raise RuntimeError(f"hidden_in must be 1D, got shape={tuple(hidden_in.shape)}")
    if hidden_prenorm.ndim != 1:
        raise RuntimeError(
            f"hidden_prenorm must be 1D, got shape={tuple(hidden_prenorm.shape)}"
        )
    if attention_output.ndim != 1:
        raise RuntimeError(
            f"attention_output must be 1D, got shape={tuple(attention_output.shape)}"
        )

    if hidden_in.shape != hidden_prenorm.shape:
        raise RuntimeError(
            f"hidden_in/hidden_prenorm shape mismatch: "
            f"{tuple(hidden_in.shape)} vs {tuple(hidden_prenorm.shape)}"
        )
    if attention_output.shape != hidden_in.shape:
        raise RuntimeError(
            f"attention_output/hidden_in shape mismatch: "
            f"{tuple(attention_output.shape)} vs {tuple(hidden_in.shape)}"
        )

    if not np.isfinite(hidden_in).all():
        raise RuntimeError("hidden_in contains non-finite values")
    if not np.isfinite(hidden_prenorm).all():
        raise RuntimeError("hidden_prenorm contains non-finite values")
    if not np.isfinite(attention_output).all():
        raise RuntimeError("attention_output contains non-finite values")

    required_aux = [
        "q_flash",
        "blocked_k_token",
    ]
    for key in required_aux:
        if key not in aux:
            raise RuntimeError(f"runtime aux missing required key: {key}")
        x = to_numpy_f32(aux[key])
        print_stats(f"runtime.attention.{key}", np.squeeze(x))
        if not np.isfinite(x).all():
            raise RuntimeError(f"runtime aux {key} contains non-finite values")

    print(
        "[smoke] attention runtime OK "
        f"layer_id={args.layer_id} position_id={args.position_id} "
        f"aux_keys={sorted(aux.keys())}"
    )


if __name__ == "__main__":
    main()
