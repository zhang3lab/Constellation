from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class ModelExecResult:
    """
    Generic execution result.

    For block-level APIs, `output` means the block contribution.
    For black-box segment APIs, `output` means the segment output hidden.
    The exact meaning is defined by the API that returns it.
    """
    output: np.ndarray
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionSharedSegmentResult:
    """
    Result for a fused attention + shared-expert segment.

    - attention_output: attention block contribution
    - shared_expert_output: shared-expert block contribution computed after
      applying the attention residual update
    """
    attention_output: np.ndarray
    shared_expert_output: np.ndarray
    aux: dict[str, Any] = field(default_factory=dict)


class FullModelRef(Protocol):
    def run_attention_block(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        position_ids=None,
        attention_mask=None,
        kv_cache=None,
        return_aux: bool = False,
    ) -> ModelExecResult:
        ...

    def run_dense_ffn_block(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        return_aux: bool = False,
    ) -> ModelExecResult:
        ...

    def run_shared_expert_block(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        return_aux: bool = False,
    ) -> ModelExecResult:
        ...

    def run_attention_shared_segment(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        position_ids=None,
        attention_mask=None,
        kv_cache=None,
        return_aux: bool = False,
    ) -> AttentionSharedSegmentResult:
        ...

    def run_prefix_segment(
        self,
        hidden_in: np.ndarray,
        *,
        start_layer: int,
        end_layer: int,
        position_ids=None,
        attention_mask=None,
        kv_cache=None,
        return_aux: bool = False,
    ) -> ModelExecResult:
        ...


class FullModelRefBase:
    """
    Thin mixin-style base class for FullModelRef implementations.

    Primitive block methods must be implemented by subclasses.
    Composite segment methods may be implemented by subclasses, or inherited
    when a generic composition is valid.
    """

    def run_attention_block(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        position_ids=None,
        attention_mask=None,
        kv_cache=None,
        return_aux: bool = False,
    ) -> ModelExecResult:
        raise NotImplementedError

    def run_dense_ffn_block(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        return_aux: bool = False,
    ) -> ModelExecResult:
        raise NotImplementedError

    def run_shared_expert_block(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        return_aux: bool = False,
    ) -> ModelExecResult:
        raise NotImplementedError

    def run_attention_shared_segment(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        position_ids=None,
        attention_mask=None,
        kv_cache=None,
        return_aux: bool = False,
    ) -> AttentionSharedSegmentResult:
        raise NotImplementedError

    def run_prefix_segment(
        self,
        hidden_in: np.ndarray,
        *,
        start_layer: int,
        end_layer: int,
        position_ids=None,
        attention_mask=None,
        kv_cache=None,
        return_aux: bool = False,
    ) -> ModelExecResult:
        raise NotImplementedError
