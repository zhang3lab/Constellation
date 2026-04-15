import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_ROOT = {
    "log_level": 0,
}

DEFAULT_RUN = {
    "mode": "validation",
    "start_layer": 0,
    "end_layer": 60,
    "collect_per_layer": True,
    "experts_per_layer": 256,
    "sparse_layer_start": 3,
    "sparse_layer_end": 60,
    "allow_drop_non_target_residents": False,
}

DEFAULT_KV_CACHE = {
    "max_batch_size": 1,
    "max_seq_len": 256,
    "page_size": 128,
    "use_triton": True,
}


def _require_object(obj: dict, key: str) -> dict:
    value = obj.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config must contain an object field '{key}'")
    return value


def _require_list(obj: dict, key: str) -> list:
    value = obj.get(key)
    if not isinstance(value, list):
        raise ValueError(f"config must contain a list field '{key}'")
    return value


def _require_nonempty_str(obj: dict, key: str, *, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}.{key} must be a non-empty string")
    return value


def _require_int(obj: dict, key: str, *, where: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{where}.{key} must be an integer")
    return value


def _require_number(obj: dict, key: str, *, where: str) -> float | int:
    value = obj.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{where}.{key} must be a number")
    return value


def _merge_defaults(obj: dict, key: str, defaults: dict) -> dict:
    value = obj.get(key)
    if value is None:
        merged = dict(defaults)
    else:
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be an object when provided")
        merged = dict(defaults)
        merged.update(value)
    obj[key] = merged
    return merged


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("config root must be a JSON object")

    merged_root = dict(DEFAULT_ROOT)
    merged_root.update(obj)
    obj = merged_root

    log_level = obj.get("log_level")
    if not isinstance(log_level, int):
        raise ValueError("log_level must be an integer")
    if log_level < 0 or log_level > 2:
        raise ValueError("log_level must be in [0, 2]")

    nodes = _require_list(obj, "nodes")
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"nodes[{i}] must be an object")
        _require_nonempty_str(node, "host", where=f"nodes[{i}]")
        _require_int(node, "control_port", where=f"nodes[{i}]")

    model = _require_object(obj, "model")
    _require_nonempty_str(model, "name", where="model")
    _require_nonempty_str(model, "family", where="model")
    _require_nonempty_str(model, "root", where="model")
    _require_int(model, "chunk_size", where="model")
    _require_int(model, "expert_mem_bytes", where="model")
    _require_number(model, "memory_utilization", where="model")

    run_cfg = _merge_defaults(obj, "run", DEFAULT_RUN)
    mode = _require_nonempty_str(run_cfg, "mode", where="run")
    allowed_modes = {
        "validation",
        "demo",
        "partial_61layer_debug",
        "full_model_debug",
    }
    if mode not in allowed_modes:
        raise ValueError(
            "run.mode must be one of: validation, demo, partial_61layer_debug, full_model_debug"
        )

    _require_int(run_cfg, "num_experts", where="run")

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

    experts_per_layer = _require_int(run_cfg, "experts_per_layer", where="run")
    if experts_per_layer <= 0:
        raise ValueError("run.experts_per_layer must be > 0")

    sparse_layer_start = _require_int(run_cfg, "sparse_layer_start", where="run")
    sparse_layer_end = _require_int(run_cfg, "sparse_layer_end", where="run")

    if sparse_layer_start < 0:
        raise ValueError("run.sparse_layer_start must be >= 0")
    if sparse_layer_end < 0:
        raise ValueError("run.sparse_layer_end must be >= 0")
    if sparse_layer_end < sparse_layer_start:
        raise ValueError("run.sparse_layer_end must be >= run.sparse_layer_start")

    start_layer = _require_int(run_cfg, "start_layer", where="run")
    end_layer = _require_int(run_cfg, "end_layer", where="run")
    if start_layer < 0:
        raise ValueError("run.start_layer must be >= 0")
    if end_layer < 0:
        raise ValueError("run.end_layer must be >= 0")
    if end_layer < start_layer:
        raise ValueError("run.end_layer must be >= run.start_layer")

    collect_per_layer = run_cfg.get("collect_per_layer")
    if not isinstance(collect_per_layer, bool):
        raise ValueError("run.collect_per_layer must be a boolean")

    allow_drop_non_target_residents = run_cfg.get("allow_drop_non_target_residents")
    if not isinstance(allow_drop_non_target_residents, bool):
        raise ValueError("run.allow_drop_non_target_residents must be a boolean")

    kv_cache = _merge_defaults(obj, "kv_cache", DEFAULT_KV_CACHE)
    max_batch_size = _require_int(kv_cache, "max_batch_size", where="kv_cache")
    max_seq_len = _require_int(kv_cache, "max_seq_len", where="kv_cache")
    page_size = _require_int(kv_cache, "page_size", where="kv_cache")

    use_triton = kv_cache.get("use_triton")
    if not isinstance(use_triton, bool):
        raise ValueError("kv_cache.use_triton must be a boolean")

    if max_batch_size <= 0:
        raise ValueError("kv_cache.max_batch_size must be > 0")
    if max_seq_len <= 0:
        raise ValueError("kv_cache.max_seq_len must be > 0")
    if page_size <= 0:
        raise ValueError("kv_cache.page_size must be > 0")

    return obj
