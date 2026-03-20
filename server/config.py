import json
from pathlib import Path
from typing import Any, Dict, List


def load_nodes_config(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("config root must be a JSON object")

    nodes = obj.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("config must contain a list field 'nodes'")

    out: List[Dict[str, Any]] = []
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"nodes[{i}] must be an object")

        host = node.get("host")
        control_port = node.get("control_port")

        if not isinstance(host, str) or not host:
            raise ValueError(f"nodes[{i}].host must be a non-empty string")
        if not isinstance(control_port, int):
            raise ValueError(f"nodes[{i}].control_port must be an integer")

        out.append({
            "host": host,
            "control_port": control_port,
        })

    return out
