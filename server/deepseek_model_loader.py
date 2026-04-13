import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from server.fp8_utils import dequant_fp8_weight_blockwise


def deepseek_tensor_name(layer_id: int, expert_id: int, tensor_kind: str) -> str:
    layer_id = int(layer_id)
    expert_id = int(expert_id)
    tensor_kind = str(tensor_kind)

    kind_to_proj = {
        "w_up": "up_proj",
        "w_gate": "gate_proj",
        "w_down": "down_proj",
    }
    try:
        proj_name = kind_to_proj[tensor_kind]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor_kind: {tensor_kind}") from exc

    return f"model.layers.{layer_id}.mlp.experts.{expert_id}.{proj_name}.weight"


class DeepseekModelLoader:
    def __init__(self, model_root: str):
        self.model_root = str(model_root)
        self._weight_map = self._load_safetensors_index(self.model_root)
        self._tensor_cache: dict[str, torch.Tensor] = {}
        self._tokenizer = None

    def _load_config_json(self) -> dict:
        config_path = Path(self.model_root) / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"config.json not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            raise ValueError("config.json must decode to dict")
        return obj

    def config(self) -> dict:
        cached = getattr(self, "_config", None)
        if cached is None:
            cached = self._load_config_json()
            self._config = cached
        return cached

    def load_tokenizer(self):
        cached = self._tokenizer
        if cached is None:
            cached = AutoTokenizer.from_pretrained(
                self.model_root,
                trust_remote_code=True,
            )
            self._tokenizer = cached
        return cached

    def router_config(self) -> dict:
        cfg = self.config()

        router_cfg = {
            "n_group": int(cfg["n_group"]),
            "topk_group": int(cfg["topk_group"]),
            "top_k": int(cfg["num_experts_per_tok"]),
            "norm_topk_prob": bool(cfg["norm_topk_prob"]),
            "routed_scaling_factor": float(cfg["routed_scaling_factor"]),
            "scoring_func": str(cfg["scoring_func"]),
            "topk_method": str(cfg["topk_method"]),
            "n_routed_experts": int(cfg["n_routed_experts"]),
            "hidden_size": int(cfg["hidden_size"]),
        }

        return router_cfg

    def mla_config(self) -> dict:
        cfg = self.config()
     
        rope_scaling = cfg.get("rope_scaling")
        if not isinstance(rope_scaling, dict):
            raise ValueError("config.json missing rope_scaling")
     
        return {
            "dim": int(cfg["hidden_size"]),
            "num_heads": int(cfg["num_attention_heads"]),
            "kv_latent_rank": int(cfg["kv_lora_rank"]),
            "q_latent_rank": int(cfg["q_lora_rank"]),
            "qk_nrope_head_dim": int(cfg["qk_nope_head_dim"]),
            "qk_rope_head_dim": int(cfg["qk_rope_head_dim"]),
            "v_head_dim": int(cfg["v_head_dim"]),
            "max_seq_len": int(cfg["max_position_embeddings"]),
            "max_seq_len_train": int(rope_scaling["original_max_position_embeddings"]),
            "beta_fast": float(rope_scaling.get("beta_fast", 32)),
            "beta_slow": float(rope_scaling.get("beta_slow", 1)),
            "rope_theta": float(cfg.get("rope_theta", 10000.0)),
            "rope_factor": float(rope_scaling.get("factor", 40.0)),
            "mscale": float(rope_scaling.get("mscale", 1.0)),
            "mscale_all_dim": float(rope_scaling.get("mscale_all_dim", 1.0)),
        }

    @staticmethod
    def _load_safetensors_index(model_root: str) -> Dict[str, str]:
        index_path = Path(model_root) / "model.safetensors.index.json"
        with index_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        weight_map = obj.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("model.safetensors.index.json missing weight_map")

        out: Dict[str, str] = {}
        for tensor_name, shard_relpath in weight_map.items():
            if not isinstance(tensor_name, str):
                raise ValueError("invalid tensor name in weight_map")
            if not isinstance(shard_relpath, str):
                raise ValueError(f"invalid shard path for tensor {tensor_name}")
            out[tensor_name] = str(Path(model_root) / shard_relpath)
        return out

    def weight_map(self) -> Dict[str, str]:
        return self._weight_map

    def resolve_tensor(self, tensor_name: str) -> Tuple[str, str]:
        tensor_name = str(tensor_name)
        shard_path = self._weight_map.get(tensor_name)
        if shard_path is None:
            raise KeyError(f"tensor not found in index: {tensor_name}")
        return tensor_name, shard_path

    def resolve_deepseek_tensor(
        self,
        layer_id: int,
        expert_id: int,
        tensor_kind: str,
    ) -> Tuple[str, str]:
        tensor_name = deepseek_tensor_name(
            layer_id=layer_id,
            expert_id=expert_id,
            tensor_kind=tensor_kind,
        )
        return self.resolve_tensor(tensor_name)

    def resolve_deepseek_scale_tensor(
        self,
        layer_id: int,
        expert_id: int,
        tensor_kind: str,
    ) -> Tuple[str, str]:
        weight_name, _shard_path = self.resolve_deepseek_tensor(
            layer_id=layer_id,
            expert_id=expert_id,
            tensor_kind=tensor_kind,
        )
        scale_name = weight_name + "_scale_inv"
        return self.resolve_tensor(scale_name)

    @staticmethod
    def load_tensor_from_open_shard(
        shard_file,
        tensor_name: str,
    ):
        t = shard_file.get_tensor(tensor_name)
        t = t.contiguous()
        tensor_bytes = t.view(torch.uint8).cpu().numpy().tobytes()
        shape = tuple(int(x) for x in t.shape)
        dtype = str(t.dtype)
        return tensor_bytes, shape, dtype

    def open_shard(self, shard_path: str):
        return safe_open(shard_path, framework="pt", device="cpu")

    def load_tensor_fp32_by_name(self, tensor_name: str) -> torch.Tensor:
        tensor_name = str(tensor_name)

        cached = self._tensor_cache.get(tensor_name)
        if cached is not None:
            return cached

        _, shard_path = self.resolve_tensor(tensor_name)

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            t = f.get_tensor(tensor_name)

            if t.dtype == torch.float8_e4m3fn:
                scale_name = tensor_name + "_scale_inv"
                if scale_name in self._weight_map:
                    _, scale_shard_path = self.resolve_tensor(scale_name)
                    with safe_open(scale_shard_path, framework="pt", device="cpu") as sf:
                        scale_inv = sf.get_tensor(scale_name).to(torch.float32).contiguous()
                    t = dequant_fp8_weight_blockwise(t, scale_inv).to(torch.float32).contiguous()
                else:
                    raise RuntimeError(f"missing scale tensor for fp8 weight: {scale_name}")
            else:
                t = t.to(torch.float32).contiguous()

        self._tensor_cache[tensor_name] = t
        return t

    def load_routed_expert_triplet_fp32(self, layer_id: int, expert_id: int):
        layer_id = int(layer_id)
        expert_id = int(expert_id)

        base = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
        w_up = self.load_tensor_fp32_by_name(f"{base}.up_proj.weight")
        w_gate = self.load_tensor_fp32_by_name(f"{base}.gate_proj.weight")
        w_down = self.load_tensor_fp32_by_name(f"{base}.down_proj.weight")
        return w_up, w_gate, w_down

    def load_shared_expert_triplet_fp32(self, layer_id: int):
        layer_id = int(layer_id)

        base = f"model.layers.{layer_id}.mlp.shared_experts"
        w_up = self.load_tensor_fp32_by_name(f"{base}.up_proj.weight")
        w_gate = self.load_tensor_fp32_by_name(f"{base}.gate_proj.weight")
        w_down = self.load_tensor_fp32_by_name(f"{base}.down_proj.weight")
        return w_up, w_gate, w_down

    def load_dense_ffn_triplet_fp32(self, layer_id: int):
        layer_id = int(layer_id)

        base = f"model.layers.{layer_id}.mlp"
        w_up = self.load_tensor_fp32_by_name(f"{base}.up_proj.weight")
        w_gate = self.load_tensor_fp32_by_name(f"{base}.gate_proj.weight")
        w_down = self.load_tensor_fp32_by_name(f"{base}.down_proj.weight")
        return w_up, w_gate, w_down

    def load_input_layernorm_weight_fp32(self, layer_id: int):
        layer_id = int(layer_id)
        return self.load_tensor_fp32_by_name(
            f"model.layers.{layer_id}.input_layernorm.weight"
        )

    def load_post_attention_layernorm_weight_fp32(self, layer_id: int):
        layer_id = int(layer_id)
        return self.load_tensor_fp32_by_name(
            f"model.layers.{layer_id}.post_attention_layernorm.weight"
        )

    def load_router_tensors_fp32(self, layer_id: int):
        layer_id = int(layer_id)

        gate_weight = self.load_tensor_fp32_by_name(
            f"model.layers.{layer_id}.mlp.gate.weight"
        )
        e_score_correction_bias = self.load_tensor_fp32_by_name(
            f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias"
        )
        return gate_weight, e_score_correction_bias

    def load_attention_block_weights_fp32(self, layer_id: int):
        layer_id = int(layer_id)
        return {
            "input_layernorm": self.load_tensor_fp32_by_name(
                f"model.layers.{layer_id}.input_layernorm.weight"
            ),
            "q_a_proj": self.load_tensor_fp32_by_name(
                f"model.layers.{layer_id}.self_attn.q_a_proj.weight"
            ),
            "q_a_layernorm": self.load_tensor_fp32_by_name(
                f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight"
            ),
            "q_b_proj": self.load_tensor_fp32_by_name(
                f"model.layers.{layer_id}.self_attn.q_b_proj.weight"
            ),
            "kv_a_proj_with_mqa": self.load_tensor_fp32_by_name(
                f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"
            ),
            "kv_a_layernorm": self.load_tensor_fp32_by_name(
                f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"
            ),
            "kv_b_proj": self.load_tensor_fp32_by_name(
                f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"
            ),
            "o_proj": self.load_tensor_fp32_by_name(
                f"model.layers.{layer_id}.self_attn.o_proj.weight"
            ),
        }

    def load_embed_tokens_weight_fp32(self):
        return self.load_tensor_fp32_by_name("model.embed_tokens.weight")

    def load_norm_weight_fp32(self):
        return self.load_tensor_fp32_by_name("model.norm.weight")

    def load_lm_head_weight_fp32(self):
        return self.load_tensor_fp32_by_name("lm_head.weight")
