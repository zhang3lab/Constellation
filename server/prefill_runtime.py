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
    input_ids: torch.Tensor,
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

    if input_ids.ndim == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids[0]

    hidden_in = executor.embed_token_ids(input_ids)

    position_ids = _build_prefill_position_ids(input_ids)
    attention_mask = _build_prefill_attention_mask(int(input_ids.numel()))

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

    final_hidden = result["output"]
    last_hidden = final_hidden[-1]

    logits_result = executor.run_final_norm_and_lm_head(
        last_hidden,
        return_aux=False,
    )
    next_token_logits = logits_result.output

    return PrefillResult(
        prompt_tokens=int(input_ids.numel()),
        next_token_logits=next_token_logits,
        next_position=int(input_ids.numel()),
        aux={
            "input_ids": input_ids,
            "final_hidden": final_hidden,
            "last_hidden": last_hidden,
            "final_position": int(input_ids.numel()) - 1,
            "per_layer": result.get("per_layer"),
            "raw_result": result,
        },
    )


def _validate_prefill_args(
    *,
    input_ids: torch.Tensor,
    start_layer: int,
    end_layer: int,
) -> None:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(f"input_ids expected torch.Tensor, got {type(input_ids).__name__}")

    if input_ids.ndim == 2 and input_ids.shape[0] == 1:
        pass
    elif input_ids.ndim != 1:
        raise RuntimeError(
            f"input_ids expected shape [T] or [1, T], got {tuple(input_ids.shape)}"
        )

    if input_ids.numel() <= 0:
        raise RuntimeError("prefill requires non-empty input_ids")

    if int(start_layer) < 0:
        raise RuntimeError(f"start_layer must be >= 0, got {start_layer}")

    if int(end_layer) < int(start_layer):
        raise RuntimeError(
            f"end_layer must be >= start_layer, got start={start_layer} end={end_layer}"
        )


def _build_prefill_position_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.arange(int(input_ids.numel()), dtype=torch.int64)


def _build_prefill_attention_mask(num_tokens: int) -> Any:
    if num_tokens <= 0:
        raise RuntimeError(f"prefill requires positive num_tokens, got {num_tokens}")

    # Current attention runtime handles the causal prefill mask internally.
    return None

