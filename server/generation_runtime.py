from __future__ import annotations

import time
from typing import Sequence

import torch

from .decode_runtime import run_decode_step_logits
from .generation_types import (
    FinishReason,
    GenerationResult,
    GenerationState,
)
from .prefill_runtime import run_prefill, run_prefill_from_input_ids
from .sample_runtime import run_sample


def _match_stop_token_sequence(
    generated_token_ids: Sequence[int],
    stop_token_sequences: Sequence[Sequence[int]],
) -> list[int] | None:
    for seq in stop_token_sequences:
        if not seq:
            continue

        seq_len = len(seq)
        if seq_len > len(generated_token_ids):
            continue

        if list(generated_token_ids[-seq_len:]) == list(seq):
            return [int(x) for x in seq]

    return None


def _commit_sampled_token(
    state: GenerationState,
    *,
    token_id: int,
) -> None:
    token_id = int(token_id)
    state.generated_token_ids.append(token_id)
    state.last_token_id = token_id


def _sync_last_token_id(state: GenerationState) -> None:
    state.last_token_id = (
        int(state.generated_token_ids[-1]) if state.generated_token_ids else None
    )


def _finish_generation(
    state: GenerationState,
    *,
    finish_reason: FinishReason,
) -> None:
    state.is_finished = True
    state.finish_reason = finish_reason


def _build_generation_result(
    session,
    *,
    state: GenerationState,
    generate_started_at: float | None,
    generate_finished_at: float | None,
    prefill_time_ms: float | None,
) -> GenerationResult:
    executor = session.full_model_executor
    if executor is None:
        raise RuntimeError("session.full_model_executor is not initialized")

    output_token_ids = [int(x) for x in state.generated_token_ids]
    output_text = executor.decode(output_token_ids) if output_token_ids else ""

    return GenerationResult(
        request_id=state.request_id,
        model_name=state.model_name,
        output_token_ids=output_token_ids,
        output_text=output_text,
        finish_reason=state.finish_reason,
        prompt_tokens=state.prompt_tokens_count,
        completion_tokens=state.completion_tokens_count,
        total_tokens=state.total_tokens_count,
        generate_started_at=generate_started_at,
        generate_finished_at=generate_finished_at,
        prefill_time_ms=prefill_time_ms,
    )


def _maybe_finish_after_commit(
    session,
    *,
    state: GenerationState,
    token_id: int,
) -> FinishReason | None:
    executor = session.full_model_executor
    if executor is None:
        raise RuntimeError("session.full_model_executor is not initialized")

    eos_token_id = getattr(executor, "eos_token_id", None)
    if eos_token_id is not None and int(token_id) == int(eos_token_id):
        return "eos_token"

    matched = _match_stop_token_sequence(
        state.generated_token_ids,
        state.sampling_config.stop_token_sequences,
    )
    if matched is not None:
        del state.generated_token_ids[-len(matched):]
        _sync_last_token_id(state)
        return "stop_token"

    return None


def run_generation_from_input_ids(
    session,
    *,
    state: GenerationState,
    input_ids: torch.Tensor,
    start_layer: int,
    end_layer: int,
    kv_cache,
) -> GenerationResult:
    if not isinstance(state, GenerationState):
        raise TypeError(
            f"state expected GenerationState, got {type(state).__name__}"
        )

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(
            f"input_ids expected torch.Tensor, got {type(input_ids).__name__}"
        )

    if kv_cache is None:
        raise RuntimeError("kv_cache must be initialized before run_generation_from_input_ids()")

    executor = session.full_model_executor
    if executor is None:
        raise RuntimeError("session.full_model_executor is not initialized")

    state.reset_for_new_generation()

    generate_started_at = time.time()
    prefill_t0 = time.perf_counter()

    prefill_result = run_prefill_from_input_ids(
        session,
        input_ids=input_ids,
        start_layer=int(start_layer),
        end_layer=int(end_layer),
        kv_cache=kv_cache,
        collect_per_layer=bool(state.debug_config.collect_per_layer),
    )

    prefill_t1 = time.perf_counter()
    prefill_time_ms = (prefill_t1 - prefill_t0) * 1000.0

    state.prompt_token_ids = input_ids
    state.last_logits = prefill_result.next_token_logits
    state.is_prefilled = True

    if state.sampling_config.max_new_tokens <= 0:
        _finish_generation(state, finish_reason="max_new_tokens")
        return _build_generation_result(
            session,
            state=state,
            generate_started_at=generate_started_at,
            generate_finished_at=time.time(),
            prefill_time_ms=prefill_time_ms,
        )

    sample_result = run_sample(
        prefill_result.next_token_logits,
        sampling_config=state.sampling_config,
    )
    token_id = int(sample_result.token_id)

    while True:
        token_position = state.next_position
        _commit_sampled_token(state, token_id=token_id)

        finish_reason = _maybe_finish_after_commit(
            session,
            state=state,
            token_id=token_id,
        )
        if finish_reason is not None:
            _finish_generation(state, finish_reason=finish_reason)
            break

        if state.completion_tokens_count >= state.sampling_config.max_new_tokens:
            _finish_generation(state, finish_reason="max_new_tokens")
            break

        current_hidden = executor.embed_token_ids(
            torch.tensor([token_id], dtype=torch.long)
        )

        decode_result = run_decode_step_logits(
            session,
            current_hidden=current_hidden,
            position_id=token_position,
            start_layer=int(start_layer),
            end_layer=int(end_layer),
            kv_cache=kv_cache,
            collect_per_layer=bool(state.debug_config.collect_per_layer),
        )
        state.last_logits = decode_result.logits

        sample_result = run_sample(
            decode_result.logits,
            sampling_config=state.sampling_config,
        )
        token_id = int(sample_result.token_id)

    return _build_generation_result(
        session,
        state=state,
        generate_started_at=generate_started_at,
        generate_finished_at=time.time(),
        prefill_time_ms=prefill_time_ms,
    )


def run_generation(
    session,
    *,
    state: GenerationState,
    prompt: str,
    start_layer: int,
    end_layer: int,
    kv_cache,
) -> GenerationResult:
    if not isinstance(prompt, str):
        raise TypeError(f"prompt expected str, got {type(prompt).__name__}")

    tokenizer = session.get_deepseek_model_loader().load_tokenizer()
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )

    if not hasattr(encoded, "get"):
        raise RuntimeError(
            f'tokenizer(..., return_tensors="pt") expected mapping-like result, got {type(encoded).__name__}'
        )
     
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise RuntimeError(
            f'tokenizer(..., return_tensors="pt") result missing torch.Tensor "input_ids", got {type(input_ids).__name__}'
        )
     
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError(
            f'input_ids expected shape [1, T], got {tuple(input_ids.shape)}'
        )

    return run_generation_from_input_ids(
        session,
        state=state,
        input_ids=input_ids,
        start_layer=int(start_layer),
        end_layer=int(end_layer),
        kv_cache=kv_cache,
    )
