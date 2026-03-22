import json
from pathlib import Path
from typing import Dict, Tuple


def deepseek_tensor_name(layer_id: int, expert_id: int, tensor_kind: str) -> str:
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


def load_safetensors_index(model_root: str) -> Dict[str, str]:
    index_path = Path(model_root) / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    weight_map = obj.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json missing weight_map")
    return weight_map


def resolve_deepseek_tensor_file(
    model_root: str,
    layer_id: int,
    expert_id: int,
    tensor_kind: str,
) -> Tuple[str, str]:
    tensor_name = deepseek_tensor_name(
        layer_id=layer_id,
        expert_id=expert_id,
        tensor_kind=tensor_kind,
    )
    weight_map = load_safetensors_index(model_root)

    shard_relpath = weight_map.get(tensor_name)
    if not isinstance(shard_relpath, str):
        raise KeyError(f"tensor not found in index: {tensor_name}")

    shard_path = str(Path(model_root) / shard_relpath)
    return tensor_name, shard_path
