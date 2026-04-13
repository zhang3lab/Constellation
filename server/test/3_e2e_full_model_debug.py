from __future__ import annotations

import argparse

import numpy as np
import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.generation_types import PrefillResult
from server.inference_session import InferenceSession
from server.prefill_runtime import run_prefill
from server.test.utils import print_stats, to_numpy_f32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--start-layer", type=int, default=None)
    ap.add_argument("--end-layer", type=int, default=None)
    ap.add_argument("--collect-per-layer", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"], log_level=cfg["log_level"])
    setup_control_plane(coord, cfg)

    run_cfg = cfg["run"]
    kv_cache_cfg = cfg["kv_cache"]

    start_layer = int(args.start_layer if args.start_layer is not None else run_cfg["start_layer"])
    end_layer = int(args.end_layer if args.end_layer is not None else run_cfg["end_layer"])
    collect_per_layer = bool(args.collect_per_layer or run_cfg["collect_per_layer"])

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)

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

        prefill_result = run_prefill(
            session,
            prompt=args.prompt,
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache=session.page_attention_cache_managers,
            collect_per_layer=collect_per_layer,
        )

        if not isinstance(prefill_result, PrefillResult):
            raise TypeError(
                f"run_prefill(...) expected PrefillResult, got {type(prefill_result).__name__}"
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
            input_ids = input_ids[0]
        elif input_ids.ndim != 1:
            raise RuntimeError(
                f"prefill_result.aux['input_ids'] expected shape [T] or [1, T], got {tuple(input_ids.shape)}"
            )
        if input_ids.numel() <= 0:
            raise RuntimeError("prefill_result.aux['input_ids'] must be non-empty")

        final_hidden_t = aux.get("final_hidden")
        last_hidden_t = aux.get("last_hidden")
        if not isinstance(final_hidden_t, torch.Tensor):
            raise TypeError(
                f"prefill_result.aux['final_hidden'] expected torch.Tensor, got {type(final_hidden_t).__name__}"
            )
        if not isinstance(last_hidden_t, torch.Tensor):
            raise TypeError(
                f"prefill_result.aux['last_hidden'] expected torch.Tensor, got {type(last_hidden_t).__name__}"
            )
        if not isinstance(prefill_result.next_token_logits, torch.Tensor):
            raise TypeError(
                f"prefill_result.next_token_logits expected torch.Tensor, got {type(prefill_result.next_token_logits).__name__}"
            )

        if final_hidden_t.ndim != 2:
            raise RuntimeError(
                f"final_hidden expected shape [T, H], got {tuple(final_hidden_t.shape)}"
            )
        if last_hidden_t.ndim != 1:
            raise RuntimeError(
                f"last_hidden expected shape [H], got {tuple(last_hidden_t.shape)}"
            )
        if final_hidden_t.shape[0] != int(input_ids.numel()):
            raise RuntimeError(
                f"final_hidden length mismatch: T={final_hidden_t.shape[0]} len(input_ids)={int(input_ids.numel())}"
            )
        if prefill_result.prompt_tokens != int(input_ids.numel()):
            raise RuntimeError(
                f"prompt_tokens mismatch: result={prefill_result.prompt_tokens} len(input_ids)={int(input_ids.numel())}"
            )
        if prefill_result.next_position != int(input_ids.numel()):
            raise RuntimeError(
                f"next_position mismatch: result={prefill_result.next_position} len(input_ids)={int(input_ids.numel())}"
            )

        final_hidden = to_numpy_f32(final_hidden_t)
        last_hidden = to_numpy_f32(last_hidden_t)
        logits = to_numpy_f32(prefill_result.next_token_logits)

        print(f"[e2e] prompt={args.prompt!r}")
        print(f"[e2e] input_ids={input_ids.tolist()}")
        print(
            f"[e2e] start_layer={start_layer} "
            f"end_layer={end_layer} prompt_tokens={prefill_result.prompt_tokens}"
        )
        print(f"[e2e] next_position={prefill_result.next_position}")

        print_stats("e2e.final_hidden", final_hidden)
        print_stats("e2e.last_hidden", last_hidden)
        print_stats("e2e.next_token_logits", logits)

        if not np.isfinite(final_hidden).all():
            raise RuntimeError("[e2e] final_hidden contains non-finite values")
        if not np.isfinite(last_hidden).all():
            raise RuntimeError("[e2e] last_hidden contains non-finite values")
        if not np.isfinite(logits).all():
            raise RuntimeError("[e2e] next_token_logits contains non-finite values")

        if not np.allclose(last_hidden, final_hidden[-1]):
            raise RuntimeError(
                "[e2e] last_hidden does not match the last row of final_hidden"
            )

        topk = int(args.topk)
        if topk <= 0:
            raise RuntimeError(f"topk must be positive, got {topk}")

        top_ids = np.argsort(logits)[-topk:][::-1]
        top_vals = logits[top_ids]
        top_tokens = [
            session.full_model_executor.decode([int(tok_id)])
            for tok_id in top_ids.tolist()
        ]

        print("[e2e] final_hidden_last[:8] =", last_hidden[:8])
        print("[e2e] logits_top_ids =", top_ids.tolist())
        print("[e2e] logits_top_vals =", top_vals.tolist())
        print("[e2e] logits_top_tokens =", top_tokens)

        if collect_per_layer:
            per_layer = aux.get("per_layer")
            if per_layer is None:
                raise RuntimeError("collect_per_layer=True but aux['per_layer'] is None")
            print(f"[e2e] collected_layers={len(per_layer)}")

        assert top_ids.shape == (topk,)
        assert len(top_tokens) == topk

        print("[e2e] full_model_debug passed")


if __name__ == "__main__":
    main()
