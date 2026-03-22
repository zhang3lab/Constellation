import json
from pathlib import Path
from typing import Dict, Tuple


def deepseek_tensor_name(expert_id: int, tensor_kind: str) -> str:
    # DeepSeek Expert: w1=up, w2=down, w3=gate
    kind_to_weight = {
        "w_up": "w1",
        "w_down": "w2",
        "w_gate": "w3",
    }
    try:
        weight_name = kind_to_weight[tensor_kind]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor_kind: {tensor_kind}") from exc

    # Common Hugging Face naming for DeepSeek-style MoE experts.
    return f"model.layers.0.mlp.experts.{expert_id}.{weight_name}.weight"


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
    expert_id: int,
    tensor_kind: str,
) -> Tuple[str, str]:
    tensor_name = deepseek_tensor_name(expert_id, tensor_kind)
    weight_map = load_safetensors_index(model_root)

    shard_relpath = weight_map.get(tensor_name)
    if not isinstance(shard_relpath, str):
        raise KeyError(f"tensor not found in index: {tensor_name}")

    shard_path = str(Path(model_root) / shard_relpath)
    return tensor_name, shard_path
