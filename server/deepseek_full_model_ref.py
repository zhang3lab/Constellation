from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from server.array_utils import as_f32_1d
from server.fp8_utils import dequant_fp8_weight_blockwise
from server.full_model_ref import (
    AttentionSharedSegmentResult,
    FullModelRefBase,
    ModelExecResult,
)


def _rms_norm(hidden: np.ndarray, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = as_f32_1d(hidden, "rms_norm.hidden")
    x_t = torch.from_numpy(x)

    rms = torch.rsqrt(torch.mean(x_t * x_t) + eps)
    y = x_t * rms
    y = y * weight
    return y.to(torch.float32)


def _post_attention_ffn_input(
    session,
    hidden: np.ndarray,
    layer_id: int,
) -> np.ndarray:
    hidden = as_f32_1d(hidden, "post_attention_ffn_input.hidden")
    layer_id = int(layer_id)

    model_loader = session.get_deepseek_model_loader()
    norm_weight = model_loader.load_post_attention_layernorm_weight_fp32(layer_id)

    y = _rms_norm(hidden, norm_weight)
    y = y.to(torch.float32).cpu().numpy()
    y = as_f32_1d(y, f"post_attention_ffn_input.layer{layer_id}")
    return y


class DeepseekFullModelExecutorBase(FullModelRefBase):
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



class DeepseekFullModelExecutor(DeepseekFullModelExecutorBase):
    def __init__(self, session):
        self.session = session


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
        hidden_in = as_f32_1d(hidden_in, "attention.hidden_in")
        layer_id = int(layer_id)
     
        if self.session.backbone_store is None:
            raise RuntimeError("session.backbone_store is not initialized")
        if self.session.attention_runtime is None:
            raise RuntimeError("session.attention_runtime is not initialized")
        if self.session.freq_cis_by_device is None:
            raise RuntimeError("session.freq_cis_by_device is not initialized")
     
        if kv_cache is None:
            raise RuntimeError("kv_cache is required for run_attention_block")
     
        cache_manager = kv_cache[layer_id]
     
        layer_entry = self.session.backbone_store.layer(layer_id)
        dev = str(layer_entry["device"])
        runtime_dtype = self.session.backbone_store.dtype
        freq_cis_all = self.session.freq_cis_by_device[dev]
     
        x = (
            torch.from_numpy(hidden_in)
            .to(device=dev, dtype=runtime_dtype)
            .view(1, 1, -1)
        )
     
        start_pos = 0 if position_ids is None else int(np.asarray(position_ids).reshape(-1)[0])
        freq_cis = freq_cis_all[start_pos : start_pos + 1]
     
        y = self.session.attention_runtime.forward(
            x,
            start_pos=start_pos,
            freq_cis=freq_cis,
            weights=layer_entry["attention"],
            cache_manager=cache_manager,
            mask=None if attention_mask is None else attention_mask,
        )
     
        out_np = y[0, 0].detach().float().cpu().numpy().astype(np.float32, copy=False)
        out_np = as_f32_1d(out_np, f"attention.layer{layer_id}.output")
     
        aux = {}
        if return_aux:
            aux = {
                "kind": "attention_block",
                "impl": "mla_runtime_triton",
                "layer_id": layer_id,
                "device": dev,
                "start_pos": start_pos,
            }
     
        return ModelExecResult(output=out_np, aux=aux)


    def run_dense_ffn_block(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        return_aux: bool = False,
    ) -> ModelExecResult:
        hidden_in = as_f32_1d(hidden_in, "dense_ffn.hidden_in")
        layer_id = int(layer_id)
     
        model_loader = self.session.get_deepseek_model_loader()
        norm_weight = model_loader.load_post_attention_layernorm_weight_fp32(layer_id)
        w_up, w_gate, w_down = model_loader.load_dense_ffn_triplet_fp32(layer_id)
     
        x_t = _rms_norm(hidden_in, norm_weight)
        up = w_up @ x_t
        gate = w_gate @ x_t
        fused = up * F.silu(gate)
        out = w_down @ fused
     
        out_np = out.to(torch.float32).cpu().numpy()
        out_np = as_f32_1d(out_np, f"dense_ffn.layer{layer_id}.output")
     
        aux = {}
        if return_aux:
            aux = {
                "kind": "dense_ffn_block",
                "impl": "torch_ref",
                "layer_id": layer_id,
            }
     
        return ModelExecResult(output=out_np, aux=aux)


    def run_shared_expert_block(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        return_aux: bool = False,
    ) -> ModelExecResult:
        hidden_in = as_f32_1d(hidden_in, "shared_expert.hidden_in")
        layer_id = int(layer_id)
     
        model_loader = self.session.get_deepseek_model_loader()
        w_up, w_gate, w_down = model_loader.load_shared_expert_triplet_fp32(layer_id)
     
        x_t = torch.from_numpy(hidden_in)
        up = w_up @ x_t
        gate = w_gate @ x_t
        fused = up * F.silu(gate)
        out = w_down @ fused
     
        out_np = out.to(torch.float32).cpu().numpy()
        out_np = as_f32_1d(out_np, f"shared_expert.layer{layer_id}.output")
     
        aux = {}
        if return_aux:
            aux = {
                "kind": "shared_expert_block",
                "impl": "torch_ref",
                "layer_id": layer_id,
            }
     
        return ModelExecResult(output=out_np, aux=aux)


    def tokenize(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
    ):
        model_loader = self.session.get_deepseek_model_loader()
        tok = model_loader.load_tokenizer()
        return tok(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )


    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        model_loader = self.session.get_deepseek_model_loader()
        tok = model_loader.load_tokenizer()
        ids = tok.encode(text, add_special_tokens=add_special_tokens)
        return [int(x) for x in ids]


    def prepare_prompt_hidden_input(self, prompt: str) -> dict:
        input_ids = self.encode(prompt)
        if not input_ids:
            raise RuntimeError("prompt encoded to empty input_ids")

        model_loader = self.session.get_deepseek_model_loader()
        embed = model_loader.load_embed_tokens_weight_fp32()

        last_id = int(input_ids[-1])
        hidden = embed[last_id].detach().cpu().numpy().astype(np.float32, copy=False)

        return {
            "prompt": prompt,
            "input_ids": input_ids,
            "hidden_in": hidden,
            "position_ids": None,
            "attention_mask": None,
            "kv_cache": None,
        }

    def run_final_norm_and_lm_head(
        self,
        hidden_in: np.ndarray,
        *,
        return_aux: bool = False,
    ) -> ModelExecResult:
        hidden_in = as_f32_1d(hidden_in, "final_head.hidden_in")
     
        if self.session.backbone_store is None:
            raise RuntimeError("session.backbone_store is not initialized")
     
        norm_w = self.session.backbone_store.model_norm()
        lm_head_w = self.session.backbone_store.lm_head()
        dev = str(norm_w.device)
        runtime_dtype = self.session.backbone_store.dtype
     
        x = torch.from_numpy(hidden_in).to(device=dev, dtype=runtime_dtype).view(1, -1)
     
        x = torch.nn.functional.rms_norm(
            x,
            (x.shape[-1],),
            norm_w,
            1e-6,
        )
        logits = torch.matmul(x, lm_head_w.t())
     
        out_np = logits[0].detach().float().cpu().numpy().astype(np.float32, copy=False)
     
        aux = {}
        if return_aux:
            aux = {
                "kind": "final_norm_and_lm_head",
                "device": dev,
            }
     
        return ModelExecResult(output=out_np, aux=aux)

    def decode_token_ids(self, token_ids) -> list[str]:
        model_loader = self.session.get_deepseek_model_loader()
        tokenizer = model_loader.load_tokenizer()
        ids = [int(x) for x in token_ids]
        return [tokenizer.decode([x]) for x in ids]
