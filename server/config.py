import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("config root must be a JSON object")

    nodes = obj.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("config must contain a list field 'nodes'")

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"nodes[{i}] must be an object")

        host = node.get("host")
        control_port = node.get("control_port")

        if not isinstance(host, str) or not host:
            raise ValueError(f"nodes[{i}].host must be a non-empty string")
        if not isinstance(control_port, int):
            raise ValueError(f"nodes[{i}].control_port must be an integer")

    model = obj.get("model")
    if not isinstance(model, dict):
        raise ValueError("config must contain an object field 'model'")

    if not isinstance(model.get("family"), str) or not model["family"]:
        raise ValueError("model.family must be a non-empty string")
    if not isinstance(model.get("root"), str) or not model["root"]:
        raise ValueError("model.root must be a non-empty string")
    if not isinstance(model.get("chunk_size"), int):
        raise ValueError("model.chunk_size must be an integer")
    if not isinstance(model.get("expert_mem_bytes"), int):
        raise ValueError("model.expert_mem_bytes must be an integer")
    if not isinstance(model.get("memory_utilization"), (int, float)):
        raise ValueError("model.memory_utilization must be a number")

    run_cfg = obj.get("run")
    if not isinstance(run_cfg, dict):
        raise ValueError("config must contain an object field 'run'")

    if not isinstance(run_cfg.get("mode"), str) or not run_cfg["mode"]:
        raise ValueError("run.mode must be a non-empty string")
    if run_cfg["mode"] not in ("validation", "demo"):
        raise ValueError("run.mode must be either 'validation' or 'demo'")
    if not isinstance(run_cfg.get("layer_id"), int):
        raise ValueError("run.layer_id must be an integer")
    if not isinstance(run_cfg.get("num_experts"), int):
        raise ValueError("run.num_experts must be an integer")

    return obj
