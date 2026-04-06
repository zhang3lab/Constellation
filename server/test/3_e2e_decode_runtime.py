from __future__ import annotations

import argparse

import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.decode_runtime import run_decode_step_logits
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.generation_types import (
    DecodeStepResult,
    GreedySampling,
    PrefillResult,
    SampleResult,
    SamplingConfig,
)
from server.inference_session import InferenceSession
from server.prefill_runtime import run_prefill
from server.sample_runtime import run_sample


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

        prefill_result = run_prefill(
            session,
            prompt=args.prompt,
            start_layer=int(args.start_layer),
            end_layer=int(args.end_layer),
            kv_cache=session.page_attention_cache_managers,
            collect_per_layer=bool(args.collect_per_layer),
        )

        if not isinstance(prefill_result, PrefillResult):
            raise TypeError(
                f"run_prefill(...) expected PrefillResult, got {type(prefill_result).__name__}"
            )

        prefill_aux = prefill_result.aux
        if prefill_aux is None:
            raise RuntimeError("prefill result aux must not be None")

        input_ids = prefill_aux.get("input_ids")
        if not isinstance(input_ids, list) or not input_ids:
            raise RuntimeError("prefill aux['input_ids'] must be a non-empty list[int]")
        if not all(isinstance(x, int) for x in input_ids):
            raise RuntimeError("prefill aux['input_ids'] must be list[int]")

        prefill_last_hidden = _require_tensor(
            prefill_aux.get("last_hidden"),
            "prefill_aux['last_hidden']",
        )
        if prefill_last_hidden.ndim != 1:
            raise RuntimeError(
                f"prefill_aux['last_hidden'] expected shape [H], got {tuple(prefill_last_hidden.shape)}"
            )
        if not torch.all(torch.isfinite(prefill_last_hidden)).item():
            raise RuntimeError("non-finite prefill_aux['last_hidden']")

        next_token_logits = _require_tensor(
            prefill_result.next_token_logits,
            "prefill_result.next_token_logits",
        )
        if next_token_logits.ndim != 1:
            raise RuntimeError(
                f"prefill_result.next_token_logits expected shape [V], got {tuple(next_token_logits.shape)}"
            )
        if not torch.all(torch.isfinite(next_token_logits)).item():
            raise RuntimeError("non-finite prefill_result.next_token_logits")

        sampling_config = SamplingConfig(strategy=GreedySampling())
        sample_result = run_sample(
            prefill_result.next_token_logits,
            sampling_config=sampling_config,
        )
        if not isinstance(sample_result, SampleResult):
            raise TypeError(
                f"run_sample(...) expected SampleResult, got {type(sample_result).__name__}"
            )

        token_id = int(sample_result.token_id)
        if token_id < 0:
            raise RuntimeError(f"sampled token_id must be >= 0, got {token_id}")

        token_position = prefill_result.next_position
        if token_position != len(input_ids):
            raise RuntimeError(
                f"decode start position mismatch: token_position={token_position} len(input_ids)={len(input_ids)}"
            )

        current_hidden = session.full_model_executor.embed_token_ids([token_id])
        current_hidden = _require_tensor(current_hidden, "executor.embed_token_ids([token_id])")

        if current_hidden.ndim == 1:
            pass
        elif current_hidden.ndim == 2 and current_hidden.shape[0] == 1:
            pass
        else:
            raise RuntimeError(
                f"decode input current_hidden expected shape [H] or [1, H], got {tuple(current_hidden.shape)}"
            )

        if not torch.all(torch.isfinite(current_hidden)).item():
            raise RuntimeError("non-finite current_hidden before decode step")

        decode_result = run_decode_step_logits(
            session,
            current_hidden=current_hidden,
            position_id=int(token_position),
            start_layer=int(args.start_layer),
            end_layer=int(args.end_layer),
            kv_cache=session.page_attention_cache_managers,
            collect_per_layer=bool(args.collect_per_layer),
        )

        if not isinstance(decode_result, DecodeStepResult):
            raise TypeError(
                f"run_decode_step_logits(...) expected DecodeStepResult, got {type(decode_result).__name__}"
            )

        decode_aux = decode_result.aux
        if decode_aux is None:
            raise RuntimeError("decode result aux must not be None")

        logits = _require_tensor(decode_result.logits, "decode_result.logits")
        final_hidden = _require_tensor(decode_aux.get("final_hidden"), "decode_aux['final_hidden']")
        last_hidden = _require_tensor(decode_aux.get("last_hidden"), "decode_aux['last_hidden']")

        if decode_result.next_position != token_position + 1:
            raise RuntimeError(
                f"next_position mismatch: result={decode_result.next_position} expected={token_position + 1}"
            )

        if logits.ndim != 1:
            raise RuntimeError(
                f"decode_result.logits expected shape [V], got {tuple(logits.shape)}"
            )
        if logits.numel() <= 0:
            raise RuntimeError("decode_result.logits must be non-empty")

        if final_hidden.ndim == 2:
            if final_hidden.shape[0] != 1:
                raise RuntimeError(
                    f"decode_aux['final_hidden'] expected shape [1, H] or [H], got {tuple(final_hidden.shape)}"
                )
            final_hidden_last = final_hidden[0]
        elif final_hidden.ndim == 1:
            final_hidden_last = final_hidden
        else:
            raise RuntimeError(
                f"decode_aux['final_hidden'] expected shape [1, H] or [H], got {tuple(final_hidden.shape)}"
            )

        if last_hidden.ndim != 1:
            raise RuntimeError(
                f"decode_aux['last_hidden'] expected shape [H], got {tuple(last_hidden.shape)}"
            )

        if not torch.all(torch.isfinite(logits)).item():
            raise RuntimeError("non-finite decode_result.logits")
        if not torch.all(torch.isfinite(final_hidden)).item():
            raise RuntimeError("non-finite decode_aux['final_hidden']")
        if not torch.all(torch.isfinite(last_hidden)).item():
            raise RuntimeError("non-finite decode_aux['last_hidden']")

        if not torch.allclose(last_hidden, final_hidden_last):
            raise RuntimeError(
                "decode_aux['last_hidden'] does not match the last token of decode_aux['final_hidden']"
            )

        per_layer = decode_aux.get("per_layer")
        if args.collect_per_layer and per_layer is None:
            raise RuntimeError("collect_per_layer=True but decode_aux['per_layer'] is None")

        print("PASS: decode runtime e2e")
        print(f"prompt={args.prompt!r}")
        print(f"prompt_tokens={prefill_result.prompt_tokens}")
        print(f"prefill_next_position={prefill_result.next_position}")
        print(f"sampled_token_id={token_id}")
        print(f"decode_position={token_position}")
        print(f"decode_next_position={decode_result.next_position}")
        print(f"current_hidden_shape={tuple(current_hidden.shape)}")
        print(f"prefill_last_hidden_shape={tuple(prefill_last_hidden.shape)}")
        print(f"final_hidden_shape={tuple(final_hidden.shape)}")
        print(f"last_hidden_shape={tuple(last_hidden.shape)}")
        print(f"logits_shape={tuple(logits.shape)}")
        print(f"current_hidden_device={current_hidden.device}")
        print(f"prefill_last_hidden_device={prefill_last_hidden.device}")
        print(f"final_hidden_device={final_hidden.device}")
        print(f"last_hidden_device={last_hidden.device}")
        print(f"logits_device={logits.device}")
        print(f"per_layer_collected={per_layer is not None}")


if __name__ == "__main__":
    main()
