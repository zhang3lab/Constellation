from typing import Any

import numpy as np

from server.array_utils import as_f32_1d
from server.full_model_ref import (
    AttentionSharedSegmentResult,
    FullModelRefBase,
    ModelExecResult,
)


class DeepseekFullModelRefBase(FullModelRefBase):
    """
    DeepSeek-specific model-structure helpers and default composed segments.

    Assumptions for the current first version:
      - layers [0, dense_layer_count) are dense decoder layers
      - layers >= dense_layer_count are sparse decoder layers
      - run_attention_shared_segment() may be composed from attention/shared blocks
      - run_prefix_segment() may be composed from attention + dense-ffn blocks
        for the dense prefix range
    """

    def dense_layer_count(self) -> int:
        return 3

    def is_sparse_layer(self, layer_id: int) -> bool:
        return int(layer_id) >= self.dense_layer_count()

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
        hidden_in = as_f32_1d(hidden_in, "attention_shared.hidden_in")
        layer_id = int(layer_id)

        attn = self.run_attention_block(
            hidden_in,
            layer_id,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            return_aux=return_aux,
        )
        attn_out = as_f32_1d(
            np.asarray(attn.output, dtype=np.float32),
            f"attention_shared.layer{layer_id}.attention_output",
        )
        if attn_out.shape != hidden_in.shape:
            raise RuntimeError(
                f"attention output shape mismatch at layer {layer_id}: "
                f"got={attn_out.shape} expected={hidden_in.shape}"
            )

        post_attn_hidden = hidden_in + attn_out
        post_attn_hidden = as_f32_1d(
            post_attn_hidden,
            f"attention_shared.layer{layer_id}.post_attn_hidden",
        )

        shared = self.run_shared_expert_block(
            post_attn_hidden,
            layer_id,
            return_aux=return_aux,
        )
        shared_out = as_f32_1d(
            np.asarray(shared.output, dtype=np.float32),
            f"attention_shared.layer{layer_id}.shared_output",
        )
        if shared_out.shape != hidden_in.shape:
            raise RuntimeError(
                f"shared expert output shape mismatch at layer {layer_id}: "
                f"got={shared_out.shape} expected={hidden_in.shape}"
            )

        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "attention": attn.aux,
                "shared_expert": shared.aux,
                "post_attn_hidden": post_attn_hidden,
            }

        return AttentionSharedSegmentResult(
            attention_output=attn_out,
            shared_expert_output=shared_out,
            aux=aux,
        )

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
        """
        Default DeepSeek prefix implementation:
        compose dense decoder layers using attention + dense_ffn blocks.

        This is intended for the dense prefix (currently first 3 layers).
        A future fused/device implementation can override this method directly.
        """
        hidden_in = as_f32_1d(hidden_in, "prefix.hidden_in")

        start_layer = int(start_layer)
        end_layer = int(end_layer)
        if end_layer < start_layer:
            raise RuntimeError(
                f"invalid layer range: start_layer={start_layer}, end_layer={end_layer}"
            )
        if start_layer < 0:
            raise RuntimeError(f"start_layer must be >= 0, got {start_layer}")
        if end_layer >= self.dense_layer_count():
            raise RuntimeError(
                f"default DeepSeek prefix path only supports dense prefix layers, "
                f"got start_layer={start_layer}, end_layer={end_layer}, "
                f"dense_layer_count={self.dense_layer_count()}"
            )

        cur = hidden_in
        per_layer = []

        for layer_id in range(start_layer, end_layer + 1):
            attn = self.run_attention_block(
                cur,
                layer_id,
                position_ids=position_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                return_aux=return_aux,
            )
            attn_out = as_f32_1d(
                np.asarray(attn.output, dtype=np.float32),
                f"prefix.layer{layer_id}.attention_output",
            )
            if attn_out.shape != cur.shape:
                raise RuntimeError(
                    f"attention output shape mismatch at layer {layer_id}: "
                    f"got={attn_out.shape} expected={cur.shape}"
                )
            cur = cur + attn_out
            cur = as_f32_1d(cur, f"prefix.layer{layer_id}.post_attention_hidden")

            ffn = self.run_dense_ffn_block(
                cur,
                layer_id,
                return_aux=return_aux,
            )
            ffn_out = as_f32_1d(
                np.asarray(ffn.output, dtype=np.float32),
                f"prefix.layer{layer_id}.dense_ffn_output",
            )
            if ffn_out.shape != cur.shape:
                raise RuntimeError(
                    f"dense ffn output shape mismatch at layer {layer_id}: "
                    f"got={ffn_out.shape} expected={cur.shape}"
                )
            cur = cur + ffn_out
            cur = as_f32_1d(cur, f"prefix.layer{layer_id}.output")

            if return_aux:
                per_layer.append(
                    {
                        "layer_id": layer_id,
                        "attention": attn.aux,
                        "dense_ffn": ffn.aux,
                    }
                )

        aux: dict[str, Any] = {}
        if return_aux:
            aux["per_layer"] = per_layer

        return ModelExecResult(output=cur, aux=aux)


class PlaceholderDeepseekFullModelRef(DeepseekFullModelRefBase):
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
        hidden_in = as_f32_1d(hidden_in, "placeholder_attention.hidden_in")
        out = np.zeros_like(hidden_in, dtype=np.float32)
        aux = {}
        if return_aux:
            aux = {
                "kind": "attention_block",
                "impl": "placeholder",
                "layer_id": int(layer_id),
            }
        return ModelExecResult(output=out, aux=aux)

    def run_dense_ffn_block(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        return_aux: bool = False,
    ) -> ModelExecResult:
        hidden_in = as_f32_1d(hidden_in, "placeholder_dense_ffn.hidden_in")
        out = np.zeros_like(hidden_in, dtype=np.float32)
        aux = {}
        if return_aux:
            aux = {
                "kind": "dense_ffn_block",
                "impl": "placeholder",
                "layer_id": int(layer_id),
            }
        return ModelExecResult(output=out, aux=aux)

    def run_shared_expert_block(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        return_aux: bool = False,
    ) -> ModelExecResult:
        hidden_in = as_f32_1d(hidden_in, "placeholder_shared_expert.hidden_in")
        out = np.zeros_like(hidden_in, dtype=np.float32)
        aux = {}
        if return_aux:
            aux = {
                "kind": "shared_expert_block",
                "impl": "placeholder",
                "layer_id": int(layer_id),
            }
        return ModelExecResult(output=out, aux=aux)
