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
    if run_cfg["mode"] not in ("validation", "demo", "partial_61layer_debug", "full_model_debug"):
        raise ValueError(
            "run.mode must be one of: validation, demo, partial_61layer_debug, full_model_debug"
        )

    if not isinstance(run_cfg.get("num_experts"), int):
        raise ValueError("run.num_experts must be an integer")

    restricted = run_cfg.get("restricted_expert_ids")
    if restricted is not None:
        if not isinstance(restricted, list):
            raise ValueError("run.restricted_expert_ids must be a list when provided")
        for i, x in enumerate(restricted):
            if not isinstance(x, int):
                raise ValueError(f"run.restricted_expert_ids[{i}] must be an integer")
            if x < 0:
                raise ValueError(
                    f"run.restricted_expert_ids[{i}] must be >= 0, got {x}"
                )

    experts_per_layer = run_cfg.get("experts_per_layer")
    if experts_per_layer is not None:
        if not isinstance(experts_per_layer, int):
            raise ValueError("run.experts_per_layer must be an integer when provided")
        if experts_per_layer <= 0:
            raise ValueError("run.experts_per_layer must be > 0")

    sparse_layer_start = run_cfg.get("sparse_layer_start")
    sparse_layer_end = run_cfg.get("sparse_layer_end")

    if sparse_layer_start is not None:
        if not isinstance(sparse_layer_start, int):
            raise ValueError("run.sparse_layer_start must be an integer when provided")
        if sparse_layer_start < 0:
            raise ValueError("run.sparse_layer_start must be >= 0")

    if sparse_layer_end is not None:
        if not isinstance(sparse_layer_end, int):
            raise ValueError("run.sparse_layer_end must be an integer when provided")
        if sparse_layer_end < 0:
            raise ValueError("run.sparse_layer_end must be >= 0")

    if sparse_layer_start is not None and sparse_layer_end is not None:
        if sparse_layer_end < sparse_layer_start:
            raise ValueError(
                "run.sparse_layer_end must be >= run.sparse_layer_start"
            )

    return obj
