from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class RestrictedRoutingSpec:
    resident_local_expert_ids: list[int]
    top_k: int
    renormalize: bool = True


def _to_resident_mask(
    selected_expert_ids: torch.Tensor,
    *,
    resident_local_expert_ids: list[int],
) -> torch.Tensor:
    if not isinstance(selected_expert_ids, torch.Tensor):
        raise TypeError(
            f"selected_expert_ids expected torch.Tensor, got {type(selected_expert_ids).__name__}"
        )
    if selected_expert_ids.ndim != 2:
        raise RuntimeError(
            f"selected_expert_ids expected shape [T, K], got {tuple(selected_expert_ids.shape)}"
        )

    resident_set = {int(x) for x in resident_local_expert_ids}
    if not resident_set:
        raise RuntimeError("resident_local_expert_ids must be non-empty")

    mask = torch.zeros_like(selected_expert_ids, dtype=torch.bool)
    for expert_id in resident_set:
        mask |= (selected_expert_ids == int(expert_id))
    return mask


def restrict_router_output(
    router_out: dict,
    *,
    resident_local_expert_ids: list[int],
    top_k: int,
    renormalize: bool,
) -> dict:
    """
    Restrict full-router outputs to a resident expert subset.

    Expected router_out:
      - router_out["selected_expert_ids"]: Tensor[T, K]
      - router_out["selected_weights"]: Tensor[T, K]

    Returns:
      {
        "selected_expert_ids": Tensor[T, top_k],
        "selected_weights": Tensor[T, top_k],
        "selected_mask": Tensor[T, top_k],
      }

    Semantics:
      - Keep only experts that are in resident_local_expert_ids.
      - Preserve original ordering among kept experts.
      - Require each token to keep at least one expert.
      - If fewer than top_k experts remain for a token, pad expert_ids with -1 and weights with 0.
      - If renormalize=True, renormalize kept weights per token to sum to 1.
    """
    if not isinstance(router_out, dict):
        raise TypeError(f"router_out expected dict, got {type(router_out).__name__}")

    selected_expert_ids = router_out.get("selected_expert_ids")
    selected_weights = router_out.get("selected_weights")

    if not isinstance(selected_expert_ids, torch.Tensor):
        raise TypeError(
            f'router_out["selected_expert_ids"] expected torch.Tensor, got {type(selected_expert_ids).__name__}'
        )
    if not isinstance(selected_weights, torch.Tensor):
        raise TypeError(
            f'router_out["selected_weights"] expected torch.Tensor, got {type(selected_weights).__name__}'
        )

    if selected_expert_ids.ndim != 2:
        raise RuntimeError(
            f"selected_expert_ids expected shape [T, K], got {tuple(selected_expert_ids.shape)}"
        )
    if selected_weights.ndim != 2:
        raise RuntimeError(
            f"selected_weights expected shape [T, K], got {tuple(selected_weights.shape)}"
        )
    if tuple(selected_expert_ids.shape) != tuple(selected_weights.shape):
        raise RuntimeError(
            f"router output shape mismatch: ids={tuple(selected_expert_ids.shape)} "
            f"weights={tuple(selected_weights.shape)}"
        )

    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    T, K = selected_expert_ids.shape
    resident_mask = _to_resident_mask(
        selected_expert_ids,
        resident_local_expert_ids=resident_local_expert_ids,
    )

    out_ids = torch.full(
        (T, top_k),
        fill_value=-1,
        dtype=selected_expert_ids.dtype,
        device=selected_expert_ids.device,
    )
    out_weights = torch.zeros(
        (T, top_k),
        dtype=selected_weights.dtype,
        device=selected_weights.device,
    )
    out_mask = torch.zeros(
        (T, top_k),
        dtype=torch.bool,
        device=selected_expert_ids.device,
    )

    for t in range(T):
        kept_ids = selected_expert_ids[t][resident_mask[t]]
        kept_weights = selected_weights[t][resident_mask[t]]

        if kept_ids.numel() == 0:
            raise RuntimeError(
                f"restricted routing removed all experts for token index {t}"
            )

        kept_ids = kept_ids[:top_k]
        kept_weights = kept_weights[:top_k]

        if renormalize:
            denom = kept_weights.sum()
            if not torch.isfinite(denom).item() or float(denom.item()) <= 0.0:
                raise RuntimeError(
                    f"restricted routing got non-positive weight sum at token index {t}"
                )
            kept_weights = kept_weights / denom

        n = int(kept_ids.numel())
        out_ids[t, :n] = kept_ids
        out_weights[t, :n] = kept_weights
        out_mask[t, :n] = True

    return {
        "selected_expert_ids": out_ids,
        "selected_weights": out_weights,
        "selected_mask": out_mask,
    }


def compare_restricted_moe_layer_vs_ref(
    session,
    *,
    prompt: str,
    layer_id: int,
    routing_spec: RestrictedRoutingSpec,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> dict:
    """
    Contract only for now.

    Intended semantics:
      1. Build the shared layer input hidden for the chosen layer_id.
      2. Run full router once on reference side.
      3. Apply restrict_router_output(...) with routing_spec.
      4. Run runtime MoE with the same resident experts and same restricted routing semantics.
      5. Compare:
         - selected expert ids
         - selected weights
         - merged expert output
         - final layer output

    This function should compare against a restricted reference model,
    not against the unrestricted full model.
    """
    raise NotImplementedError
