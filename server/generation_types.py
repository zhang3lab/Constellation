from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal


KV_BACKEND_PAGED: Final[str] = "paged"
KV_BACKEND_CONTIGUOUS: Final[str] = "contiguous"
KV_BACKEND_EXTERNAL: Final[str] = "external"


KvBackend = Literal[
    "paged",
    "contiguous",
    "external",
]


FinishReason = Literal[
    "eos_token",
    "stop_token",
    "stop_string",
    "max_new_tokens",
]


class SamplingStrategy:
    pass


@dataclass(slots=True)
class GreedySampling(SamplingStrategy):
    pass


@dataclass(slots=True)
class TemperatureTopPSampling(SamplingStrategy):
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass(slots=True)
class SamplingConfig:
    strategy: SamplingStrategy = field(default_factory=GreedySampling)
    max_new_tokens: int = 16
    stop_token_ids: list[int] = field(default_factory=list)
    stop_strings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DebugConfig:
    collect_per_layer: bool = False
    return_logits: bool = False
    expert_trace: bool = False
    timing_trace: bool = False


@dataclass(slots=True)
class KvCacheHandle:
    backend: KvBackend
    handle: Any | None = None
    capacity_tokens: int | None = None
    page_size: int | None = None
    num_layers: int | None = None
    dtype: str | None = None


@dataclass(slots=True)
class GenerationState:
    request_id: str
    model_name: str

    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    debug_config: DebugConfig = field(default_factory=DebugConfig)

    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)

    kv_cache: KvCacheHandle | None = None
    last_logits: Any | None = None
    last_token_id: int | None = None

    is_prefilled: bool = False
    is_finished: bool = False
    finish_reason: FinishReason | None = None

    def reset_for_new_generation(self) -> None:
        self.prompt_token_ids.clear()
        self.generated_token_ids.clear()
        self.kv_cache = None
        self.last_logits = None
        self.last_token_id = None
        self.is_prefilled = False
        self.is_finished = False
        self.finish_reason = None

    @property
    def prompt_tokens_count(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def completion_tokens_count(self) -> int:
        return len(self.generated_token_ids)

    @property
    def total_tokens_count(self) -> int:
        return self.prompt_tokens_count + self.completion_tokens_count

    @property
    def can_decode(self) -> bool:
        return (
            self.is_prefilled
            and (not self.is_finished)
            and (self.last_logits is not None)
        )


@dataclass(slots=True)
class PrefillResult:
    prompt_tokens: int
    prefill_time_ms: float | None = None
    per_layer_stats: Any | None = None


@dataclass(slots=True)
class DecodeStepResult:
    token_id: int
    text: str | None = None
    finish_reason: FinishReason | None = None
    decode_time_ms: float | None = None


@dataclass(slots=True)
class GenerationResult:
    request_id: str
    model_name: str

    output_token_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    finish_reason: FinishReason | None = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    generate_started_at: float | None = None
    generate_finished_at: float | None = None

    prefill_time_ms: float | None = None
