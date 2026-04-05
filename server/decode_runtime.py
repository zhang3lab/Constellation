from __future__ import annotations

from typing import Any

import torch

from server.array_utils import ARRCFG_HIDDEN_TORCH, as_array, torch_dtype_name
from server.full_model_runtime import run_full_model
from server.generation_types import DecodeStepResult


def _validate_decode_current_hidden(
    current_hidden: Any,
    *,
    position_id: int,
) -> torch.Tensor:
    if not isinstance(current_hidden, torch.Tensor):
        raise TypeError(
            f"current_hidden expected torch.Tensor, got {type(current_hidden).__name__}"
        )

    if current_hidden.ndim == 1:
        out = current_hidden
    elif current_hidden.ndim == 2 and current_hidden.shape[0] == 1:
        out = current_hidden[0]
    else:
        raise RuntimeError(
            f"decode current_hidden expected shape [H] or [1, H], "
            f"got {tuple(current_hidden.shape)}"
        )

    if not torch.all(torch.isfinite(out)).item():
        raise RuntimeError(f"non-finite current_hidden at decode position {position_id}")

    return out


def run_decode_step_logits(
    session,
    *,
    current_hidden,
    position_id: int,
    start_layer: int,
    end_layer: int,
    kv_cache,
    collect_per_layer: bool = False,
) -> DecodeStepResult:
    executor = session.full_model_executor
    if executor is None:
        raise RuntimeError("session.full_model_executor is not initialized")

    if not isinstance(position_id, int):
        raise TypeError(f"position_id expected int, got {type(position_id).__name__}")

    if int(start_layer) < 0:
        raise RuntimeError(f"start_layer must be >= 0, got {start_layer}")

    if int(end_layer) < int(start_layer):
        raise RuntimeError(
            f"end_layer must be >= start_layer, got start={start_layer} end={end_layer}"
        )

    current_hidden = _validate_decode_current_hidden(
        current_hidden,
        position_id=position_id,
    )

    position_ids = torch.tensor(
        [position_id],
        dtype=torch.int64,
        device=current_hidden.device,
    )

    result = run_full_model(
        session,
        current_hidden,
        start_layer=int(start_layer),
        end_layer=int(end_layer),
        position_ids=position_ids,
        attention_mask=None,
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

    if final_hidden.ndim == 2:
        if final_hidden.shape[0] != 1:
            raise RuntimeError(
                f"decode expected single-token final_hidden, got shape={tuple(final_hidden.shape)}"
            )
        last_hidden = final_hidden[0]
    elif final_hidden.ndim == 1:
        last_hidden = final_hidden
    else:
        raise RuntimeError(
            f"decode expected final_hidden shape [H] or [1, H], got {tuple(final_hidden.shape)}"
        )

    if not torch.all(torch.isfinite(last_hidden)).item():
        raise RuntimeError(f"non-finite last_hidden at decode position {position_id}")

    logits_result = executor.run_final_norm_and_lm_head(
        last_hidden,
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

    return DecodeStepResult(
        logits=logits,
        next_position=int(position_id) + 1,
        aux={
            "final_hidden": final_hidden,
            "last_hidden": last_hidden,
            "per_layer": result.get("per_layer"),
            "raw_result": result,
        },
    )
