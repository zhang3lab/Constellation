import json
import torch
from pathlib import Path
from typing import Dict, Tuple

from safetensors import safe_open


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


def load_tensor_bytes_from_safetensors(
    shard_path: str,
    tensor_name: str,
):
    import torch
    from safetensors import safe_open

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        t = f.get_tensor(tensor_name)

    t = t.contiguous()
    tensor_bytes = t.view(torch.uint8).cpu().numpy().tobytes()
    shape = tuple(int(x) for x in t.shape)
    dtype = str(t.dtype)
    return tensor_bytes, shape, dtype


def resolve_and_load_deepseek_tensor(
    model_root: str,
    layer_id: int,
    expert_id: int,
    tensor_kind: str,
) -> Tuple[str, str, bytes, Tuple[int, ...], str]:
    tensor_name, shard_path = resolve_deepseek_tensor_file(
        model_root=model_root,
        layer_id=layer_id,
        expert_id=expert_id,
        tensor_kind=tensor_kind,
    )

    tensor_bytes, shape, dtype = load_tensor_bytes_from_safetensors(
        shard_path=shard_path,
        tensor_name=tensor_name,
    )

    row_block = 128
    col_block = 128
    return tensor_name, shard_path, tensor_bytes, shape, dtype, row_block, col_block

def resolve_and_load_deepseek_scale_tensor(
    model_root: str,
    layer_id: int,
    expert_id: int,
    tensor_kind: str,
) -> tuple[str, str, bytes, tuple[int, ...], str]:
    weight_name, shard_path = resolve_deepseek_tensor_file(
        model_root=model_root,
        layer_id=layer_id,
        expert_id=expert_id,
        tensor_kind=tensor_kind,
    )

    scale_name = weight_name + "_scale_inv"

    tensor_bytes, shape, dtype = load_tensor_bytes_from_safetensors(
        shard_path=shard_path,
        tensor_name=scale_name,
    )

    row_block = 128
    col_block = 128
    return scale_name, shard_path, tensor_bytes, shape, dtype, row_block, col_block
