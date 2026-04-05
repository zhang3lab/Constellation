from __future__ import annotations

from typing import Any

import torch

from .full_model_runtime import run_full_model
from .generation_types import PrefillResult


def run_prefill(
    session,
    *,
    prompt: str,
    start_layer: int,
    end_layer: int,
    kv_cache,
    collect_per_layer: bool = False,
) -> PrefillResult:
    if not isinstance(prompt, str):
        raise TypeError(f"prompt expected str, got {type(prompt).__name__}")

    executor = session.full_model_executor
    if executor is None:
        raise RuntimeError("session.full_model_executor is not initialized")

    input_ids = executor.encode(prompt)
    if not input_ids:
        raise RuntimeError("prompt encoded to empty input_ids")

    return run_prefill_from_input_ids(
        session,
        input_ids=input_ids,
        start_layer=start_layer,
        end_layer=end_layer,
        kv_cache=kv_cache,
        collect_per_layer=collect_per_layer,
    )


def run_prefill_from_input_ids(
    session,
    *,
    input_ids: list[int],
    start_layer: int,
    end_layer: int,
    kv_cache,
    collect_per_layer: bool = False,
) -> PrefillResult:
    _validate_prefill_args(
        input_ids=input_ids,
        start_layer=start_layer,
        end_layer=end_layer,
    )

    executor = session.full_model_executor
    if executor is None:
        raise RuntimeError("session.full_model_executor is not initialized")

    hidden_in = executor.embed_token_ids(input_ids)
    if not isinstance(hidden_in, torch.Tensor):
        raise TypeError(
            f"executor.embed_token_ids(input_ids) expected torch.Tensor, got {type(hidden_in).__name__}"
        )

    if hidden_in.ndim != 2:
        raise RuntimeError(
            f"prefill hidden_in expected shape [T, H], got {tuple(hidden_in.shape)}"
        )

    if hidden_in.shape[0] != len(input_ids):
        raise RuntimeError(
            f"prefill hidden_in length mismatch: "
            f"T={hidden_in.shape[0]} vs len(input_ids)={len(input_ids)}"
        )

    if not torch.all(torch.isfinite(hidden_in)).item():
        raise RuntimeError("non-finite hidden_in in prefill")

    position_ids = _build_prefill_position_ids(input_ids)
    attention_mask = _build_prefill_attention_mask(len(input_ids))

    result = run_full_model(
        session,
        hidden_in,
        start_layer=int(start_layer),
        end_layer=int(end_layer),
        position_ids=position_ids,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
        collect_per_layer=bool(collect_per_layer),
    )

    if not isinstance(result, dict):
        raise TypeError(
            f"run_full_model(...) expected dict result, got {type(result).__name__}"
        )

    if "output" not in result:
        raise RuntimeError('run_full_model(...) result missing key "output"')

    final_hidden = result["output"]
    if not isinstance(final_hidden, torch.Tensor):
        raise TypeError(
            f'result["output"] expected torch.Tensor, got {type(final_hidden).__name__}'
        )

    if final_hidden.ndim != 2:
        raise RuntimeError(
            f"prefill expected final_hidden shape [T, H], got {tuple(final_hidden.shape)}"
        )

    if final_hidden.shape[0] != len(input_ids):
        raise RuntimeError(
            f"prefill output length mismatch: "
            f"T={final_hidden.shape[0]} vs len(input_ids)={len(input_ids)}"
        )

    if not torch.all(torch.isfinite(final_hidden)).item():
        raise RuntimeError("non-finite final_hidden in prefill")

    last_hidden = final_hidden[-1]
    if not isinstance(last_hidden, torch.Tensor):
        raise TypeError(
            f"final_hidden[-1] expected torch.Tensor, got {type(last_hidden).__name__}"
        )

    if last_hidden.ndim != 1:
        raise RuntimeError(
            f"prefill expected last_hidden shape [H], got {tuple(last_hidden.shape)}"
        )

    if not torch.all(torch.isfinite(last_hidden)).item():
        raise RuntimeError("non-finite last_hidden in prefill")

    logits_result = executor.run_final_norm_and_lm_head(
        last_hidden,
        return_aux=False,
    )
    if not hasattr(logits_result, "output"):
        raise RuntimeError(
            'executor.run_final_norm_and_lm_head(...) result missing attribute "output"'
        )

    next_token_logits = logits_result.output
    if not isinstance(next_token_logits, torch.Tensor):
        raise TypeError(
            f"logits_result.output expected torch.Tensor, got {type(next_token_logits).__name__}"
        )

    if next_token_logits.ndim != 1:
        raise RuntimeError(
            f"prefill expected next_token_logits shape [V], got {tuple(next_token_logits.shape)}"
        )

    if not torch.all(torch.isfinite(next_token_logits)).item():
        raise RuntimeError("non-finite next_token_logits in prefill")

    return PrefillResult(
        prompt_tokens=len(input_ids),
        next_token_logits=next_token_logits,
        next_position=len(input_ids),
        aux={
            "input_ids": [int(x) for x in input_ids],
            "final_hidden": final_hidden,
            "last_hidden": last_hidden,
            "final_position": len(input_ids) - 1,
            "per_layer": result.get("per_layer"),
            "raw_result": result,
        },
    )


def _validate_prefill_args(
    *,
    input_ids: list[int],
    start_layer: int,
    end_layer: int,
) -> None:
    if not isinstance(input_ids, list):
        raise TypeError(f"input_ids expected list[int], got {type(input_ids).__name__}")

    if not input_ids:
        raise RuntimeError("prefill requires non-empty input_ids")

    for i, x in enumerate(input_ids):
        if not isinstance(x, int):
            raise TypeError(f"input_ids[{i}] expected int, got {type(x).__name__}")

    if int(start_layer) < 0:
        raise RuntimeError(f"start_layer must be >= 0, got {start_layer}")

    if int(end_layer) < int(start_layer):
        raise RuntimeError(
            f"end_layer must be >= start_layer, got start={start_layer} end={end_layer}"
        )


def _build_prefill_position_ids(input_ids: list[int]) -> torch.Tensor:
    return torch.arange(len(input_ids), dtype=torch.int64)


def _build_prefill_attention_mask(num_tokens: int) -> Any:
    if num_tokens <= 0:
        raise RuntimeError(f"prefill requires positive num_tokens, got {num_tokens}")

    # Current attention runtime handles the causal prefill mask internally.
    return None

