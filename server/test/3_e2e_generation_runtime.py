from __future__ import annotations

import argparse
import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.generation_runtime import run_generation
from server.generation_types import (
    GenerationResult,
    GenerationState,
    GreedySampling,
    SamplingConfig,
)
from server.inference_session import InferenceSession


def _require_valid_finish_reason(x) -> str:
    valid = {"eos_token", "stop_token", "max_new_tokens"}
    if x not in valid:
        raise RuntimeError(f"invalid finish_reason: {x!r}")
    return str(x)


def _run_case(
    session,
    *,
    prompt: str,
    start_layer: int,
    end_layer: int,
    kv_cache,
    request_id: str,
    max_new_tokens: int,
) -> GenerationResult:
    state = GenerationState(
        request_id=request_id,
        model_name="deepseek-v3",
        sampling_config=SamplingConfig(
            strategy=GreedySampling(),
            max_new_tokens=int(max_new_tokens),
        ),
    )

    result = run_generation(
        session,
        state=state,
        prompt=prompt,
        start_layer=int(start_layer),
        end_layer=int(end_layer),
        kv_cache=kv_cache,
    )

    if not isinstance(result, GenerationResult):
        raise TypeError(
            f"run_generation(...) expected GenerationResult, got {type(result).__name__}"
        )

    _require_valid_finish_reason(result.finish_reason)

    if result.prompt_tokens != len(state.prompt_token_ids):
        raise RuntimeError(
            f"prompt_tokens mismatch: result={result.prompt_tokens} state={len(state.prompt_token_ids)}"
        )

    if result.completion_tokens != len(result.output_token_ids):
        raise RuntimeError(
            f"completion_tokens mismatch: result={result.completion_tokens} "
            f"len(output_token_ids)={len(result.output_token_ids)}"
        )

    if result.total_tokens != result.prompt_tokens + result.completion_tokens:
        raise RuntimeError(
            f"total_tokens mismatch: total={result.total_tokens} "
            f"prompt={result.prompt_tokens} completion={result.completion_tokens}"
        )

    if result.completion_tokens > int(max_new_tokens):
        raise RuntimeError(
            f"completion_tokens exceeds max_new_tokens: "
            f"completion={result.completion_tokens} max_new_tokens={max_new_tokens}"
        )

    if not state.is_prefilled:
        raise RuntimeError("generation state should be prefilled after run_generation()")

    if not state.is_finished:
        raise RuntimeError("generation state should be finished after run_generation()")

    if state.finish_reason != result.finish_reason:
        raise RuntimeError(
            f"finish_reason mismatch: state={state.finish_reason!r} result={result.finish_reason!r}"
        )

    if state.prompt_tokens_count != result.prompt_tokens:
        raise RuntimeError(
            f"state prompt_tokens_count mismatch: state={state.prompt_tokens_count} result={result.prompt_tokens}"
        )

    if state.completion_tokens_count != result.completion_tokens:
        raise RuntimeError(
            f"state completion_tokens_count mismatch: "
            f"state={state.completion_tokens_count} result={result.completion_tokens}"
        )

    if state.total_tokens_count != result.total_tokens:
        raise RuntimeError(
            f"state total_tokens_count mismatch: state={state.total_tokens_count} result={result.total_tokens}"
        )

    if state.generated_token_ids != result.output_token_ids:
        raise RuntimeError("state.generated_token_ids does not match result.output_token_ids")

    if result.generate_started_at is None or result.generate_finished_at is None:
        raise RuntimeError("generation timestamps must not be None")

    if result.generate_finished_at < result.generate_started_at:
        raise RuntimeError(
            "generate_finished_at must be >= generate_started_at"
        )

    if result.prefill_time_ms is None or result.prefill_time_ms < 0.0:
        raise RuntimeError(
            f"invalid prefill_time_ms: {result.prefill_time_ms!r}"
        )

    if not isinstance(result.output_text, str):
        raise RuntimeError(
            f"output_text expected str, got {type(result.output_text).__name__}"
        )

    if result.completion_tokens == 0:
        if state.last_token_id is not None:
            raise RuntimeError("state.last_token_id must be None when no tokens were generated")
    else:
        if state.last_token_id != result.output_token_ids[-1]:
            raise RuntimeError(
                f"state.last_token_id mismatch: state={state.last_token_id} "
                f"last_output={result.output_token_ids[-1]}"
            )
        if state.last_logits is None:
            raise RuntimeError("state.last_logits must not be None after non-empty generation")

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=60)
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

        session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)
        if session.page_attention_cache_managers is None:
            raise RuntimeError("session.page_attention_cache_managers is not initialized")
        if not isinstance(session.page_attention_cache_managers, dict):
            raise RuntimeError("session.page_attention_cache_managers must be a dict")

        kv_cache = session.page_attention_cache_managers

        result0 = _run_case(
            session,
            prompt=args.prompt,
            start_layer=int(args.start_layer),
            end_layer=int(args.end_layer),
            kv_cache=kv_cache,
            request_id="e2e-gen-0",
            max_new_tokens=0,
        )
        if result0.finish_reason != "max_new_tokens":
            raise RuntimeError(
                f"case max_new_tokens=0 expected finish_reason='max_new_tokens', got {result0.finish_reason!r}"
            )
        if result0.completion_tokens != 0:
            raise RuntimeError(
                f"case max_new_tokens=0 expected completion_tokens=0, got {result0.completion_tokens}"
            )

        session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)
        kv_cache = session.page_attention_cache_managers
        if kv_cache is None:
            raise RuntimeError("kv_cache re-init failed for case max_new_tokens=1")
        if not isinstance(kv_cache, dict):
            raise RuntimeError("kv_cache must be a dict after reset_full_model_kv_cache()")

        result1 = _run_case(
            session,
            prompt=args.prompt,
            start_layer=int(args.start_layer),
            end_layer=int(args.end_layer),
            kv_cache=kv_cache,
            request_id="e2e-gen-1",
            max_new_tokens=1,
        )
        if result1.completion_tokens > 1:
            raise RuntimeError(
                f"case max_new_tokens=1 expected completion_tokens <= 1, got {result1.completion_tokens}"
            )

        session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)
        kv_cache = session.page_attention_cache_managers
        if kv_cache is None:
            raise RuntimeError("kv_cache re-init failed for case max_new_tokens=4")
        if not isinstance(kv_cache, dict):
            raise RuntimeError("kv_cache must be a dict after reset_full_model_kv_cache()")

        result4 = _run_case(
            session,
            prompt=args.prompt,
            start_layer=int(args.start_layer),
            end_layer=int(args.end_layer),
            kv_cache=kv_cache,
            request_id="e2e-gen-4",
            max_new_tokens=4,
        )
        if result4.completion_tokens > 4:
            raise RuntimeError(
                f"case max_new_tokens=4 expected completion_tokens <= 4, got {result4.completion_tokens}"
            )

        print("PASS: generation runtime e2e")
        print(f"prompt={args.prompt!r}")

        print("--- case max_new_tokens=0 ---")
        print(f"finish_reason={result0.finish_reason}")
        print(f"prompt_tokens={result0.prompt_tokens}")
        print(f"completion_tokens={result0.completion_tokens}")
        print(f"total_tokens={result0.total_tokens}")
        print(f"output_text={result0.output_text!r}")

        print("--- case max_new_tokens=1 ---")
        print(f"finish_reason={result1.finish_reason}")
        print(f"prompt_tokens={result1.prompt_tokens}")
        print(f"completion_tokens={result1.completion_tokens}")
        print(f"total_tokens={result1.total_tokens}")
        print(f"output_token_ids={result1.output_token_ids}")
        print(f"output_text={result1.output_text!r}")

        print("--- case max_new_tokens=4 ---")
        print(f"finish_reason={result4.finish_reason}")
        print(f"prompt_tokens={result4.prompt_tokens}")
        print(f"completion_tokens={result4.completion_tokens}")
        print(f"total_tokens={result4.total_tokens}")
        print(f"output_token_ids={result4.output_token_ids}")
        print(f"output_text={result4.output_text!r}")


if __name__ == "__main__":
    main()
