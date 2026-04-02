from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from server.array_utils import (
    ARRCFG_HIDDEN_NUMPY_F32,
    ARRCFG_VECTOR_NUMPY_F32,
    ARRCFG_HIDDEN_TORCH,
    ARRCFG_VECTOR_TORCH,
    as_array,
    torch_dtype_name,
)
from server.fp8_utils import dequant_fp8_weight_blockwise
from server.full_model_types import (
    AttentionSharedSegmentResult,
    FullModelRefBase,
    ModelExecResult,
)


def _rms_norm(hidden, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"rms_norm.weight expected torch.Tensor, got {type(weight).__name__}")

    dtype_name = torch_dtype_name(weight.dtype)
    dev = str(weight.device)

    x = as_array(hidden, "rms_norm.hidden", ARRCFG_HIDDEN_TORCH(dtype_name, dev))
    weight = as_array(weight, "rms_norm.weight", ARRCFG_VECTOR_TORCH(dtype_name, dev))

    was_1d = (x.ndim == 1)
    x2 = x.unsqueeze(0) if was_1d else x

    rms = torch.rsqrt(torch.mean(x2 * x2, dim=-1, keepdim=True) + eps)
    y = x2 * rms
    y = y * weight.unsqueeze(0)

    if was_1d:
        y = y[0]
    return y


def _post_attention_ffn_input(
    session,
    hidden,
    layer_id: int,
):
    layer_id = int(layer_id)

    if session.backbone_store is None:
        raise RuntimeError("session.backbone_store is not initialized")

    layer_entry = session.backbone_store.layer(layer_id)
    dev = str(layer_entry["device"])
    runtime_dtype = session.backbone_store.dtype
    dtype_name = torch_dtype_name(runtime_dtype)

    norm_weight = layer_entry["post_attention_layernorm"]
    if not isinstance(norm_weight, torch.Tensor):
        raise TypeError(
            f"post_attention_ffn_input.layer{layer_id}.norm_weight expected torch.Tensor, "
            f"got {type(norm_weight).__name__}"
        )

    vector_cfg = ARRCFG_VECTOR_TORCH(dtype_name, dev)
    hidden_cfg = ARRCFG_HIDDEN_TORCH(dtype_name, dev)

    norm_weight = as_array(
        norm_weight,
        f"post_attention_ffn_input.layer{layer_id}.norm_weight",
        vector_cfg,
    )
    hidden = as_array(
        hidden,
        "post_attention_ffn_input.hidden",
        hidden_cfg,
    )

    y = _rms_norm(hidden, norm_weight)
    y = as_array(y, f"post_attention_ffn_input.layer{layer_id}", hidden_cfg)
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
        hidden_in,
        layer_id: int,
        *,
        position_ids=None,
        attention_mask=None,
        kv_cache=None,
        return_aux: bool = False,
    ) -> AttentionSharedSegmentResult:
        layer_id = int(layer_id)
     
        if self.session.backbone_store is None:
            raise RuntimeError("session.backbone_store is not initialized")
     
        layer_entry = self.session.backbone_store.layer(layer_id)
        dev = str(layer_entry["device"])
        runtime_dtype = self.session.backbone_store.dtype
        dtype_name = torch_dtype_name(runtime_dtype)
     
        hidden_cfg = ARRCFG_HIDDEN_TORCH(dtype_name, dev)
     
        hidden_in = as_array(
            hidden_in,
            "attention_shared.hidden_in",
            hidden_cfg,
        )
     
        attn = self.run_attention_block(
            hidden_in,
            layer_id,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            return_aux=return_aux,
        )
        attn = _normalize_model_exec_result(attn, f"attention_shared.layer{layer_id}.attention")
        attn_out = attn.output
     
        if tuple(attn_out.shape) != tuple(hidden_in.shape):
            raise RuntimeError(
                f"attention output shape mismatch at layer {layer_id}: "
                f"got={tuple(attn_out.shape)} expected={tuple(hidden_in.shape)}"
            )
     
        post_attn_hidden = hidden_in + attn_out
        post_attn_hidden = as_array(
            post_attn_hidden,
            f"attention_shared.layer{layer_id}.post_attn_hidden",
            hidden_cfg,
        )
     
        shared = self.run_shared_expert_block(
            post_attn_hidden,
            layer_id,
            return_aux=return_aux,
        )
        shared = _normalize_model_exec_result(
            shared,
            f"attention_shared.layer{layer_id}.shared_expert",
        )
        shared_out = shared.output
     
        if tuple(shared_out.shape) != tuple(hidden_in.shape):
            raise RuntimeError(
                f"shared expert output shape mismatch at layer {layer_id}: "
                f"got={tuple(shared_out.shape)} expected={tuple(hidden_in.shape)}"
            )
     
        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "attention": attn.aux,
                "shared_expert": shared.aux,
                "post_attn_hidden": post_attn_hidden.clone(),
            }
     
        return AttentionSharedSegmentResult(
            attention_output=attn_out,
            shared_expert_output=shared_out,
            aux=aux,
        )

    def run_prefix_segment(
        self,
        hidden_in,
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
     
        if self.session.backbone_store is None:
            raise RuntimeError("session.backbone_store is not initialized")
     
        runtime_dtype = self.session.backbone_store.dtype
        dtype_name = torch_dtype_name(runtime_dtype)
     
        first_layer_entry = self.session.backbone_store.layer(start_layer)
        first_dev = str(first_layer_entry["device"])
     
        hidden_in = as_array(
            hidden_in,
            "prefix.hidden_in",
            ARRCFG_HIDDEN_TORCH(dtype_name, first_dev),
        )
     
        cur = hidden_in
        per_layer = []
     
        for layer_id in range(start_layer, end_layer + 1):
            layer_entry = self.session.backbone_store.layer(layer_id)
            layer_dev = str(layer_entry["device"])
     
            cur = as_array(
                cur,
                f"prefix.layer{layer_id}.input",
                ARRCFG_HIDDEN_TORCH(dtype_name, layer_dev),
            )
     
            attn = self.run_attention_block(
                cur,
                layer_id,
                position_ids=position_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                return_aux=return_aux,
            )
            attn = _normalize_model_exec_result(attn, f"prefix.layer{layer_id}.attention")
            attn_out = attn.output
     
            if tuple(attn_out.shape) != tuple(cur.shape):
                raise RuntimeError(
                    f"attention output shape mismatch at layer {layer_id}: "
                    f"got={tuple(attn_out.shape)} expected={tuple(cur.shape)}"
                )
     
            cur = cur + attn_out
            cur = as_array(
                cur,
                f"prefix.layer{layer_id}.post_attention_hidden",
                ARRCFG_HIDDEN_TORCH(dtype_name, layer_dev),
            )
     
            ffn = self.run_dense_ffn_block(
                cur,
                layer_id,
                return_aux=return_aux,
            )
            ffn = _normalize_model_exec_result(ffn, f"prefix.layer{layer_id}.dense_ffn")
            ffn_out = ffn.output
     
            if tuple(ffn_out.shape) != tuple(cur.shape):
                raise RuntimeError(
                    f"dense ffn output shape mismatch at layer {layer_id}: "
                    f"got={tuple(ffn_out.shape)} expected={tuple(cur.shape)}"
                )
     
            cur = cur + ffn_out
            cur = as_array(
                cur,
                f"prefix.layer{layer_id}.output",
                ARRCFG_HIDDEN_TORCH(dtype_name, layer_dev),
            )
     
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
        hidden_in,
        layer_id: int,
        *,
        position_ids=None,
        attention_mask=None,
        kv_cache=None,
        return_aux: bool = False,
    ) -> ModelExecResult:
        layer_id = int(layer_id)
     
        if self.session.backbone_store is None:
            raise RuntimeError("session.backbone_store is not initialized")
        if self.session.attention_runtime is None:
            raise RuntimeError("session.attention_runtime is not initialized")
        if self.session.freq_cis_by_device is None:
            raise RuntimeError("session.freq_cis_by_device is not initialized")
     
        if kv_cache is None:
            raise RuntimeError("kv_cache is required for run_attention_block")
        if layer_id not in kv_cache:
            raise RuntimeError(f"kv_cache missing layer {layer_id}")
     
        cache_manager = kv_cache[layer_id]
     
        layer_entry = self.session.backbone_store.layer(layer_id)
        dev = str(layer_entry["device"])
        runtime_dtype = self.session.backbone_store.dtype
        dtype_name = torch_dtype_name(runtime_dtype)
        freq_cis_all = self.session.freq_cis_by_device[dev]
     
        hidden_cfg = ARRCFG_HIDDEN_TORCH(dtype_name, dev)
     
        hidden_in = as_array(
            hidden_in,
            "attention.hidden_in",
            hidden_cfg,
        )
        was_1d = (hidden_in.ndim == 1)
        hidden_2d = hidden_in.unsqueeze(0) if was_1d else hidden_in
     
        seq_len = int(hidden_2d.shape[0])
        if seq_len <= 0:
            raise ValueError("attention.hidden_in must have positive seq_len")
     
        x = hidden_2d.unsqueeze(0)  # [1, T, H]
     
        if position_ids is None:
            start_pos = 0
        else:
            pos_arr = np.asarray(position_ids).reshape(-1)
            if pos_arr.size == 0:
                raise ValueError("position_ids must not be empty")
            start_pos = int(pos_arr[0])
     
        if start_pos < 0:
            raise ValueError(f"start_pos must be non-negative, got {start_pos}")
     
        if not isinstance(freq_cis_all, torch.Tensor):
            raise TypeError(
                f"freq_cis_by_device[{dev}] expected torch.Tensor, got {type(freq_cis_all).__name__}"
            )
        if str(freq_cis_all.device) != dev:
            raise RuntimeError(
                f"freq_cis_by_device[{dev}] expected device={dev}, got device={freq_cis_all.device}"
            )
        if freq_cis_all.dtype != runtime_dtype:
            raise TypeError(
                f"freq_cis_by_device[{dev}] expected dtype={runtime_dtype}, got dtype={freq_cis_all.dtype}"
            )
     
        end_pos = start_pos + seq_len
        if end_pos > int(freq_cis_all.shape[0]):
            raise ValueError(
                f"freq_cis slice out of range: start_pos={start_pos}, seq_len={seq_len}, "
                f"freq_cis_len={int(freq_cis_all.shape[0])}"
            )
     
        freq_cis = freq_cis_all[start_pos:end_pos]
     
        mask = attention_mask
        if isinstance(mask, torch.Tensor):
            if str(mask.device) != dev:
                raise RuntimeError(
                    f"attention.layer{layer_id}.mask expected device={dev}, got device={mask.device}"
                )
     
        if return_aux:
            y, rt_aux = self.session.attention_runtime.forward(
                x,
                start_pos=start_pos,
                freq_cis=freq_cis,
                weights=layer_entry["attention"],
                cache_manager=cache_manager,
                mask=mask,
                return_aux=True,
            )
        else:
            y = self.session.attention_runtime.forward(
                x,
                start_pos=start_pos,
                freq_cis=freq_cis,
                weights=layer_entry["attention"],
                cache_manager=cache_manager,
                mask=mask,
                return_aux=False,
            )
            rt_aux = {}
     
        if not isinstance(y, torch.Tensor):
            raise TypeError(
                f"attention.layer{layer_id}.output expected torch.Tensor, got {type(y).__name__}"
            )
     
        out = y[0]  # [T, H]
        out = as_array(
            out,
            f"attention.layer{layer_id}.output",
            hidden_cfg,
        )
        if was_1d:
            out = out[0]
     
        aux = {}
        if return_aux:
            aux = {
                "kind": "attention_block",
                "impl": "mla_runtime_triton",
                "layer_id": layer_id,
                "device": dev,
                "start_pos": start_pos,
                "seq_len": seq_len,
                "end_pos": end_pos,
            }
            aux.update(rt_aux)
     
        return ModelExecResult(output=out, aux=aux)


    def run_dense_ffn_block(
        self,
        hidden_in,
        layer_id: int,
        *,
        return_aux: bool = False,
    ) -> ModelExecResult:
        layer_id = int(layer_id)
     
        if self.session.backbone_store is None:
            raise RuntimeError("session.backbone_store is not initialized")
     
        layer_entry = self.session.backbone_store.layer(layer_id)
        dev = str(layer_entry["device"])
        runtime_dtype = self.session.backbone_store.dtype
        dtype_name = torch_dtype_name(runtime_dtype)
     
        if "dense_ffn" not in layer_entry:
            raise RuntimeError(f"dense_ffn weights are not loaded for layer {layer_id}")
     
        dense_ffn_entry = layer_entry["dense_ffn"]
        norm_weight = layer_entry["post_attention_layernorm"]
        w_up = dense_ffn_entry["w_up"]
        w_gate = dense_ffn_entry["w_gate"]
        w_down = dense_ffn_entry["w_down"]
     
        hidden_cfg = ARRCFG_HIDDEN_TORCH(dtype_name, dev)
        vector_cfg = ARRCFG_VECTOR_TORCH(dtype_name, dev)
     
        hidden_in = as_array(hidden_in, "dense_ffn.hidden_in", hidden_cfg)
        was_1d = (hidden_in.ndim == 1)
        hidden_2d = hidden_in.unsqueeze(0) if was_1d else hidden_in
     
        norm_weight = as_array(norm_weight, f"dense_ffn.layer{layer_id}.norm_weight", vector_cfg)
        w_up = as_array(w_up, f"dense_ffn.layer{layer_id}.w_up", hidden_cfg)
        w_gate = as_array(w_gate, f"dense_ffn.layer{layer_id}.w_gate", hidden_cfg)
        w_down = as_array(w_down, f"dense_ffn.layer{layer_id}.w_down", hidden_cfg)
     
        x_t = _rms_norm(hidden_2d, norm_weight)   # (T, H)
        up = x_t @ w_up.T                         # (T, I)
        gate = x_t @ w_gate.T                     # (T, I)
        fused = up * F.silu(gate)                 # (T, I)
        out = fused @ w_down.T                    # (T, H)
     
        out = as_array(out, f"dense_ffn.layer{layer_id}.output", hidden_cfg)
        if was_1d:
            out = out[0]
     
        aux = {}
        if return_aux:
            aux = {
                "kind": "dense_ffn_block",
                "impl": "torch_runtime_vectorized",
                "layer_id": layer_id,
                "seq_len": int(hidden_2d.shape[0]),
                "device": str(out.device),
                "dtype": str(out.dtype),
            }
     
        return ModelExecResult(output=out, aux=aux)


    def run_shared_expert_block(
        self,
        hidden_in,
        layer_id: int,
        *,
        return_aux: bool = False,
    ) -> ModelExecResult:
        layer_id = int(layer_id)
     
        if self.session.backbone_store is None:
            raise RuntimeError("session.backbone_store is not initialized")
     
        layer_entry = self.session.backbone_store.layer(layer_id)
        dev = str(layer_entry["device"])
        runtime_dtype = self.session.backbone_store.dtype
     
        if "shared_expert" not in layer_entry:
            raise RuntimeError(f"shared_expert weights are not loaded for layer {layer_id}")
     
        shared_entry = layer_entry["shared_expert"]
     
        dtype_name = str(runtime_dtype).replace("torch.", "")
        hidden_cfg = ARRCFG_HIDDEN_TORCH(dtype_name, dev)
     
        hidden_in = as_array(hidden_in, "shared_expert.hidden_in", hidden_cfg)
        was_1d = (hidden_in.ndim == 1)
        hidden_2d = hidden_in.unsqueeze(0) if was_1d else hidden_in
     
        w_up = as_array(shared_entry["w_up"], f"shared_expert.layer{layer_id}.w_up", hidden_cfg)
        w_gate = as_array(shared_entry["w_gate"], f"shared_expert.layer{layer_id}.w_gate", hidden_cfg)
        w_down = as_array(shared_entry["w_down"], f"shared_expert.layer{layer_id}.w_down", hidden_cfg)
     
        up = hidden_2d @ w_up.T
        gate = hidden_2d @ w_gate.T
        fused = up * F.silu(gate)
        out = fused @ w_down.T
     
        out = as_array(out, f"shared_expert.layer{layer_id}.output", hidden_cfg)
        if was_1d:
            out = out[0]
     
        aux = {}
        if return_aux:
            aux = {
                "kind": "shared_expert_block",
                "impl": "torch_runtime_vectorized",
                "layer_id": layer_id,
                "seq_len": int(hidden_2d.shape[0]),
                "device": str(out.device),
                "dtype": str(out.dtype),
            }
     
        return ModelExecResult(output=out, aux=aux)


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


    def embed_token_id(self, token_id: int):
        token_id = int(token_id)
     
        if self.session.backbone_store is None:
            raise RuntimeError("session.backbone_store is not initialized")
     
        emb = self.session.backbone_store.embed_tokens()
        if emb is None:
            raise RuntimeError("session.backbone_store.embed_tokens is not initialized")
        if not isinstance(emb, torch.Tensor):
            raise TypeError(f"embed_tokens expected torch.Tensor, got {type(emb).__name__}")
     
        if token_id < 0 or token_id >= int(emb.shape[0]):
            raise RuntimeError(
                f"token_id out of range: got={token_id} vocab_size={int(emb.shape[0])}"
            )
     
        x = emb[token_id]
        x = as_array(
            x,
            f"embed_token_id[{token_id}]",
            ARRCFG_VECTOR_TORCH(torch_dtype_name(emb.dtype), str(emb.device)),
        )
        return x


    def prepare_prompt_hidden_input(self, prompt: str) -> dict:
        """
        Prepare prompt-side inputs for full-model execution.
     
        Return value guarantees:
        - prepared["hidden_in"]: torch.Tensor for the last prompt token input hidden
        - prepared["input_ids"]: token ids of the prompt
     
        Optional fields:
        - prepared["position_ids"]: may be None; callers may fall back to len(input_ids) - 1
        - prepared["attention_mask"]: may be None
        - prepared["kv_cache"]: may be None
        """
        input_ids = self.encode(prompt)
        if not input_ids:
            raise RuntimeError("prompt encoded to empty input_ids")
     
        last_id = int(input_ids[-1])
        hidden = self.embed_token_id(last_id)
     
        return {
            "prompt": prompt,
            "input_ids": input_ids,
            "hidden_in": hidden,
            "position_ids": None,
            "attention_mask": None,
            "kv_cache": None,
        }


    def infer_prompt_last_position(self, prepared: dict) -> int:
        prepared_pos = prepared.get("position_ids")
        if prepared_pos is not None:
            return int(np.asarray(prepared_pos).reshape(-1)[0])

        input_ids = prepared.get("input_ids")
        if input_ids is None:
            raise RuntimeError("prepared prompt data missing input_ids")
        return len(input_ids) - 1



    def run_final_norm_and_lm_head(
        self,
        hidden_in,
        *,
        return_aux: bool = False,
    ) -> ModelExecResult:
        if self.session.backbone_store is None:
            raise RuntimeError("session.backbone_store is not initialized")
     
        norm_w = self.session.backbone_store.model_norm()
        lm_head_w = self.session.backbone_store.lm_head()
     
        if norm_w is None:
            raise RuntimeError("session.backbone_store.model_norm is not initialized")
        if lm_head_w is None:
            raise RuntimeError("session.backbone_store.lm_head is not initialized")
        if not isinstance(norm_w, torch.Tensor):
            raise TypeError(f"model_norm expected torch.Tensor, got {type(norm_w).__name__}")
        if not isinstance(lm_head_w, torch.Tensor):
            raise TypeError(f"lm_head expected torch.Tensor, got {type(lm_head_w).__name__}")
     
        dev = str(norm_w.device)
        dtype_name = torch_dtype_name(norm_w.dtype)
        hidden_cfg = ARRCFG_HIDDEN_TORCH(dtype_name, dev)
     
        hidden_in = as_array(
            hidden_in,
            "final_head.hidden_in",
            hidden_cfg,
        )
        was_1d = (hidden_in.ndim == 1)
        x = hidden_in.unsqueeze(0) if was_1d else hidden_in   # [T, H] or [1, H]
     
        if str(lm_head_w.device) != dev:
            raise RuntimeError(
                f"lm_head device mismatch: norm={norm_w.device} lm_head={lm_head_w.device}"
            )
        if lm_head_w.dtype != norm_w.dtype:
            raise TypeError(
                f"lm_head dtype mismatch: norm={norm_w.dtype} lm_head={lm_head_w.dtype}"
            )
     
        x = torch.nn.functional.rms_norm(
            x,
            (x.shape[-1],),
            norm_w,
            1e-6,
        )
        logits = torch.matmul(x, lm_head_w.t())   # [T, vocab]
     
        out = as_array(
            logits[0] if was_1d else logits,
            "final_head.output",
            ARRCFG_HIDDEN_TORCH(dtype_name, dev),
        )
     
        aux = {}
        if return_aux:
            aux = {
                "kind": "final_norm_and_lm_head",
                "device": dev,
                "dtype": str(out.dtype),
            }
     
        return ModelExecResult(output=out, aux=aux)

    def decode_token_ids(self, token_ids) -> list[str]:
        model_loader = self.session.get_deepseek_model_loader()
        tokenizer = model_loader.load_tokenizer()
        ids = [int(x) for x in token_ids]
        return [tokenizer.decode([x]) for x in ids]
