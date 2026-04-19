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


class DeepseekFullModelExecutorBase(FullModelRefBase):
    """
    DeepSeek-specific model-structure helpers and default composed segments.

    Assumptions for the current first version:
      - layers [0, dense_layer_count) are dense decoder layers
      - layers >= dense_layer_count are sparse decoder layers
    """

    def dense_layer_count(self) -> int:
        return 3

    def is_sparse_layer(self, layer_id: int) -> bool:
        return int(layer_id) >= self.dense_layer_count()


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
            start_pos = int(position_ids.reshape(-1)[0].item())
     
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
        w_up = dense_ffn_entry["w_up"]
        w_gate = dense_ffn_entry["w_gate"]
        w_down = dense_ffn_entry["w_down"]
     
        hidden_cfg = ARRCFG_HIDDEN_TORCH(dtype_name, dev)
     
        hidden_in = as_array(hidden_in, "dense_ffn.hidden_in", hidden_cfg)
        was_1d = (hidden_in.ndim == 1)
        hidden_2d = hidden_in.unsqueeze(0) if was_1d else hidden_in
     
        w_up = as_array(w_up, f"dense_ffn.layer{layer_id}.w_up", hidden_cfg)
        w_gate = as_array(w_gate, f"dense_ffn.layer{layer_id}.w_gate", hidden_cfg)
        w_down = as_array(w_down, f"dense_ffn.layer{layer_id}.w_down", hidden_cfg)
     
        # hidden_in is already post_attention_layernorm(hidden)
        x_t = hidden_2d                            # (T, H)
        up = x_t @ w_up.T                          # (T, I)
        gate = x_t @ w_gate.T                      # (T, I)
        fused = up * F.silu(gate)                  # (T, I)
        out = fused @ w_down.T                     # (T, H)
     
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


    def embed_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.session.backbone_store is None:
            raise RuntimeError("session.backbone_store is not initialized")
     
        emb = self.session.backbone_store.embed_tokens()
        if emb is None:
            raise RuntimeError("session.backbone_store.embed_tokens is not initialized")
        if not isinstance(emb, torch.Tensor):
            raise TypeError(f"embed_tokens expected torch.Tensor, got {type(emb).__name__}")
     
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError(
                f"token_ids expected torch.Tensor, got {type(token_ids).__name__}"
            )
     
        if token_ids.ndim == 1:
            ids = token_ids
        elif token_ids.ndim == 2 and token_ids.shape[0] == 1:
            ids = token_ids[0]
        else:
            raise RuntimeError(
                f"token_ids expected shape [T] or [1, T], got {tuple(token_ids.shape)}"
            )
     
        if ids.numel() <= 0:
            raise RuntimeError("embed_token_ids requires non-empty token_ids")
     
        if ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(
                f"token_ids expected integer tensor dtype, got {ids.dtype}"
            )
     
        ids = ids.to(device=emb.device, dtype=torch.long)
     
        vocab_size = int(emb.shape[0])
     
        min_id = int(ids.min().item())
        max_id = int(ids.max().item())
        if min_id < 0 or max_id >= vocab_size:
            raise RuntimeError(
                f"token_ids out of range: min_id={min_id} max_id={max_id} vocab_size={vocab_size}"
            )
     
        x = emb.index_select(0, ids)
     
        x = as_array(
            x,
            f"embed_token_ids[len={int(ids.numel())}]",
            ARRCFG_HIDDEN_TORCH(torch_dtype_name(emb.dtype), str(emb.device)),
        )
        return x


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

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        model_loader = self.session.get_deepseek_model_loader()
        tokenizer = model_loader.load_tokenizer()
        ids = [int(x) for x in token_ids]
        return tokenizer.decode(ids, skip_special_tokens=bool(skip_special_tokens))
