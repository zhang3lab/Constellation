import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors import safe_open

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
        weight_name, shard_path = self.resolve_deepseek_tensor(
            layer_id=layer_id,
            expert_id=expert_id,
            tensor_kind=tensor_kind,
        )
        scale_name = weight_name + "_scale_inv"
        resolved_name, resolved_shard_path = self.resolve_tensor(scale_name)

        if resolved_shard_path != shard_path:
            raise RuntimeError(
                f"scale tensor shard mismatch for {scale_name}: "
                f"{resolved_shard_path} vs {shard_path}"
            )

        return resolved_name, resolved_shard_path

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
                keys = set(f.keys())
                if scale_name not in keys:
                    raise RuntimeError(f"missing scale tensor for fp8 weight: {scale_name}")

                scale_inv = f.get_tensor(scale_name).to(torch.float32).contiguous()
                t = dequant_fp8_weight_blockwise(t, scale_inv).to(torch.float32).contiguous()
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
