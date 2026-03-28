import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors import safe_open


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


class DeepseekModelLocator:
    def __init__(self, model_root: str):
        self.model_root = str(model_root)
        self._weight_map = self._load_safetensors_index(self.model_root)

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


def resolve_deepseek_tensor_file(
    model_root: str,
    layer_id: int,
    expert_id: int,
    tensor_kind: str,
) -> Tuple[str, str]:
    locator = DeepseekModelLocator(model_root)
    return locator.resolve_deepseek_tensor(
        layer_id=layer_id,
        expert_id=expert_id,
        tensor_kind=tensor_kind,
    )
