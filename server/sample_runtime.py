from __future__ import annotations

from typing import Any

import torch

from .generation_types import (
    GreedySampling,
    SampleResult,
    SamplingConfig,
    TemperatureTopPSampling,
)


def _validate_sampling_logits(logits: Any) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            f"logits expected torch.Tensor, got {type(logits).__name__}"
        )

    if logits.ndim != 1:
        raise RuntimeError(
            f"sampling expected 1D logits for single-token sampling, "
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

    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    work_logits = logits / temperature

    if top_p == 1.0:
        probs = torch.softmax(work_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return int(sampled.item())

    sorted_logits, sorted_indices = torch.sort(work_logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

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


def run_sample(
    logits: Any,
    *,
    sampling_config: SamplingConfig,
) -> SampleResult:
    if not isinstance(sampling_config, SamplingConfig):
        raise TypeError(
            f"sampling_config expected SamplingConfig, got {type(sampling_config).__name__}"
        )

    strategy = sampling_config.strategy

    if isinstance(strategy, GreedySampling):
        token_id = sample_greedy_from_logits(logits)
        return SampleResult(
            token_id=token_id,
            aux={
                "strategy": "greedy",
            },
        )

    if isinstance(strategy, TemperatureTopPSampling):
        token_id = sample_temperature_top_p_from_logits(
            logits,
            temperature=float(strategy.temperature),
            top_p=float(strategy.top_p),
        )
        return SampleResult(
            token_id=token_id,
            aux={
                "strategy": "temperature_top_p",
                "temperature": float(strategy.temperature),
                "top_p": float(strategy.top_p),
            },
        )

    raise TypeError(
        f"unsupported sampling strategy type: {type(strategy).__name__}"
    )
