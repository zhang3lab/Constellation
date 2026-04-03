from __future__ import annotations

from typing import Any

import numpy as np
import torch

from server.array_utils import ARRCFG_HIDDEN_TORCH, as_array, torch_dtype_name
from server.full_model_runtime import run_full_model


def _validate_sampling_logits(logits: Any) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            f"logits expected torch.Tensor, got {type(logits).__name__}"
        )

    if logits.ndim != 1:
        raise RuntimeError(
            f"sampling expected 1D logits for single-token decode, "
            f"got shape={tuple(logits.shape)}"
        )

    if not torch.all(torch.isfinite(logits)).item():
        raise RuntimeError("non-finite logits in sampling")

    return logits


def sample_greedy_from_logits(logits: Any) -> int:
    logits = _validate_sampling_logits(logits)
    return int(torch.argmax(logits).item())


def sample_temperature_top_p_from_logits(
    logits: Any,
    *,
    temperature: float,
    top_p: float,
) -> int:
    logits = _validate_sampling_logits(logits)

    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    if top_p <= 0.0:
        raise ValueError(f"top_p must be > 0, got {top_p}")

    work_logits = logits if temperature == 1.0 else (logits / temperature)

    if top_p >= 1.0:
        probs = torch.softmax(work_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return int(sampled.item())

    sorted_logits, sorted_indices = torch.sort(work_logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep the first token that crosses top_p, remove the rest.
    remove_mask = cumulative_probs > top_p
    remove_mask[1:] = remove_mask[:-1].clone()
    remove_mask[0] = False

    filtered_probs = sorted_probs.masked_fill(remove_mask, 0.0)
    prob_sum = filtered_probs.sum()

    if not torch.isfinite(prob_sum).item() or prob_sum.item() <= 0.0:
        return int(sorted_indices[0].item())

    filtered_probs = filtered_probs / prob_sum
    sampled_in_sorted = torch.multinomial(filtered_probs, num_samples=1)
    token_id = sorted_indices[sampled_in_sorted]

    return int(token_id.item())


def run_decode_step_logits(
    session,
    *,
    current_hidden,
    position_id: int,
    start_layer: int,
    end_layer: int,
    kv_cache,
):
    executor = session.full_model_executor
    if executor is None:
        raise RuntimeError("session.full_model_executor is not initialized")

    if not isinstance(position_id, int):
        raise TypeError(f"position_id expected int, got {type(position_id).__name__}")

    if not isinstance(current_hidden, torch.Tensor):
        raise TypeError(
            f"current_hidden expected torch.Tensor, got {type(current_hidden).__name__}"
        )

    if not torch.all(torch.isfinite(current_hidden)).item():
        raise RuntimeError(f"non-finite current_hidden at decode position {position_id}")

    if int(start_layer) < 0:
        raise RuntimeError(f"start_layer must be >= 0, got {start_layer}")

    if int(end_layer) < int(start_layer):
        raise RuntimeError(
            f"end_layer must be >= start_layer, got start={start_layer} end={end_layer}"
        )

    result = run_full_model(
        session,
        current_hidden,
        start_layer=int(start_layer),
        end_layer=int(end_layer),
        position_ids=np.asarray([position_id], dtype=np.int64),
        attention_mask=None,
        kv_cache=kv_cache,
        collect_per_layer=False,
    )

    final_hidden = result["output"]
    if not isinstance(final_hidden, torch.Tensor):
        raise TypeError(
            f'run_full_model(... )["output"] expected torch.Tensor, got {type(final_hidden).__name__}'
        )

    final_hidden = as_array(
        final_hidden,
        f"decode.position{position_id}.final_hidden",
        ARRCFG_HIDDEN_TORCH(
            torch_dtype_name(final_hidden.dtype),
            str(final_hidden.device),
        ),
    )
    if not torch.all(torch.isfinite(final_hidden)).item():
        raise RuntimeError(f"non-finite final_hidden at decode position {position_id}")

    logits_result = executor.run_final_norm_and_lm_head(
        final_hidden,
        return_aux=False,
    )
    logits = logits_result.output
    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            f"logits_result.output expected torch.Tensor, got {type(logits).__name__}"
        )

    logits = as_array(
        logits,
        f"decode.position{position_id}.logits",
        ARRCFG_HIDDEN_TORCH(
            torch_dtype_name(logits.dtype),
            str(logits.device),
        ),
    )
    if not torch.all(torch.isfinite(logits)).item():
        raise RuntimeError(f"non-finite logits at decode position {position_id}")

    if logits.ndim != 1:
        raise RuntimeError(
            f"decode expected 1D logits for single-token decode, got shape={tuple(logits.shape)}"
        )

    return {
        "position": int(position_id),
        "final_hidden": final_hidden,
        "logits": logits,
        "raw_result": result,
    }
