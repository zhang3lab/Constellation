from __future__ import annotations

import argparse

import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.generation_types import PrefillResult
from server.inference_session import InferenceSession
from server.prefill_runtime import run_prefill


def _require_tensor(x, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} expected torch.Tensor, got {type(x).__name__}")
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=60)
    ap.add_argument("--collect-per-layer", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    kv_cache_cfg = cfg["kv_cache"]

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

        result = run_prefill(
            session,
            prompt=args.prompt,
            start_layer=int(args.start_layer),
            end_layer=int(args.end_layer),
            kv_cache=session.page_attention_cache_managers,
            collect_per_layer=bool(args.collect_per_layer),
        )

        if not isinstance(result, PrefillResult):
            raise TypeError(
                f"run_prefill(...) expected PrefillResult, got {type(result).__name__}"
            )

        aux = result.aux
        if aux is None:
            raise RuntimeError("prefill result aux must not be None")

        input_ids = aux.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("prefill aux['input_ids'] must be torch.Tensor")
        if input_ids.ndim == 2:
            if input_ids.shape[0] != 1:
                raise RuntimeError(
                    f"prefill aux['input_ids'] expected shape [T] or [1, T], got {tuple(input_ids.shape)}"
                )
            input_ids = input_ids[0]
        elif input_ids.ndim != 1:
            raise RuntimeError(
                f"prefill aux['input_ids'] expected shape [T] or [1, T], got {tuple(input_ids.shape)}"
            )
        if input_ids.numel() <= 0:
            raise RuntimeError("prefill aux['input_ids'] must be non-empty")

        final_hidden = _require_tensor(aux.get("final_hidden"), "aux['final_hidden']")
        last_hidden = _require_tensor(aux.get("last_hidden"), "aux['last_hidden']")
        next_token_logits = _require_tensor(result.next_token_logits, "result.next_token_logits")

        if result.prompt_tokens != int(input_ids.numel()):
            raise RuntimeError(
                f"prompt_tokens mismatch: result={result.prompt_tokens} len(input_ids)={int(input_ids.numel())}"
            )

        if result.next_position != int(input_ids.numel()):
            raise RuntimeError(
                f"next_position mismatch: result={result.next_position} len(input_ids)={int(input_ids.numel())}"
            )

        final_position = aux.get("final_position")
        if final_position != int(input_ids.numel()) - 1:
            raise RuntimeError(
                f"final_position mismatch: aux={final_position} expected={int(input_ids.numel()) - 1}"
            )

        if final_hidden.ndim != 2:
            raise RuntimeError(
                f"aux['final_hidden'] expected shape [T, H], got {tuple(final_hidden.shape)}"
            )

        if final_hidden.shape[0] != int(input_ids.numel()):
            raise RuntimeError(
                f"final_hidden length mismatch: T={final_hidden.shape[0]} len(input_ids)={int(input_ids.numel())}"
            )

        if last_hidden.ndim != 1:
            raise RuntimeError(
                f"aux['last_hidden'] expected shape [H], got {tuple(last_hidden.shape)}"
            )

        if next_token_logits.ndim != 1:
            raise RuntimeError(
                f"result.next_token_logits expected shape [V], got {tuple(next_token_logits.shape)}"
            )

        if next_token_logits.numel() <= 0:
            raise RuntimeError("result.next_token_logits must be non-empty")

        if not torch.all(torch.isfinite(final_hidden)).item():
            raise RuntimeError("non-finite final_hidden in prefill test")

        if not torch.all(torch.isfinite(last_hidden)).item():
            raise RuntimeError("non-finite last_hidden in prefill test")

        if not torch.all(torch.isfinite(next_token_logits)).item():
            raise RuntimeError("non-finite next_token_logits in prefill test")

        if not torch.allclose(last_hidden, final_hidden[-1]):
            raise RuntimeError("aux['last_hidden'] does not match aux['final_hidden'][-1]")

        per_layer = aux.get("per_layer")
        if args.collect_per_layer and per_layer is None:
            raise RuntimeError("collect_per_layer=True but aux['per_layer'] is None")

        print("PASS: prefill runtime e2e")
        print(f"prompt={args.prompt!r}")
        print(f"input_ids={input_ids.tolist()}")
        print(f"prompt_tokens={result.prompt_tokens}")
        print(f"next_position={result.next_position}")
        print(f"final_position={final_position}")
        print(f"final_hidden_shape={tuple(final_hidden.shape)}")
        print(f"last_hidden_shape={tuple(last_hidden.shape)}")
        print(f"next_token_logits_shape={tuple(next_token_logits.shape)}")
        print(f"final_hidden_device={final_hidden.device}")
        print(f"last_hidden_device={last_hidden.device}")
        print(f"next_token_logits_device={next_token_logits.device}")
        print(f"per_layer_collected={per_layer is not None}")


if __name__ == "__main__":
    main()
