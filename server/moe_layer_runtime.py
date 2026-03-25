import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from server.model_locator import resolve_deepseek_tensor_file
from server.router_runtime import (
    get_router_config,
    get_router_tensors,
    route_token_real,
)
from server.test_utils import make_safe_input, print_stats, compare_arrays


def _find_target_placement(coord, expert_id: int):
    for p in coord.placements:
        if int(p["expert_id"]) == int(expert_id):
            return p
    raise RuntimeError(f"expert {expert_id} not found in placements")


def infer_one_expert(session, expert_id: int, hidden: np.ndarray):
    coord = session.coord
    target = _find_target_placement(coord, expert_id)
    host = target["host"]
    port = target["control_port"]

    pool = session.client_pool

    try:
        client = pool.get(host, port)
        resp = client.send_infer_request(
            {
                "expert_id": expert_id,
                "batch_size": 1,
                "hidden_dim": int(hidden.shape[0]),
                "activation": hidden.astype(np.float32).tobytes(),
            }
        )
    except Exception:
        pool.invalidate(host, port)
        raise

    if resp["status_code"] != 0:
        raise RuntimeError(
            f"infer failed for expert {expert_id}: "
            f"status={resp['status_code']} output_bytes={len(resp['output'])}"
        )

    y = np.frombuffer(resp["output"], dtype=np.float16).astype(np.float32)
    if y.size != hidden.shape[0]:
        raise RuntimeError(
            f"unexpected output size for expert {expert_id}: "
            f"got={y.size} expected={hidden.shape[0]}"
        )
    return y


def _load_one_weight_tensor(model_root: str, layer_id: int, expert_id: int, tensor_kind: str):
    tensor_name, shard_path = resolve_deepseek_tensor_file(
        model_root=model_root,
        layer_id=layer_id,
        expert_id=expert_id,
        tensor_kind=tensor_kind,
    )
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        t = f.get_tensor(tensor_name)
    t = t.to(torch.float32).contiguous()

    print(
        f"[top8-ref] loaded {tensor_kind}: "
        f"name={tensor_name} shape={tuple(t.shape)} dtype={t.dtype}"
    )
    return t


def validate_routes(routes):
    if not routes:
        raise RuntimeError("routes is empty")

    seen = set()
    for expert_id, weight in routes:
        expert_id = int(expert_id)
        weight = float(weight)

        if expert_id in seen:
            raise RuntimeError(f"duplicate expert_id in routes: {expert_id}")
        seen.add(expert_id)

        if weight < 0:
            raise RuntimeError(f"negative route weight for expert {expert_id}: {weight}")


def normalize_routes(routes):
    validate_routes(routes)

    total = float(sum(float(w) for _, w in routes))
    if total == 0.0:
        raise RuntimeError("route weights sum to zero")

    return [(int(eid), float(w) / total) for eid, w in routes]


def dispatch_topk_experts(session, hidden: np.ndarray, routes):
    validate_routes(routes)
    routes = [(int(eid), float(w)) for eid, w in routes]

    weighted_outputs = []
    for expert_id, weight in routes:
        y = infer_one_expert(session, expert_id, hidden)
        weighted_outputs.append((expert_id, weight, y))
    return weighted_outputs


def run_topk_moe_layer(session, hidden: np.ndarray, routes):
    weighted_outputs = dispatch_topk_experts(session, hidden, routes)
    combined = combine_outputs(weighted_outputs)
    return combined, weighted_outputs


def run_moe_layer(session, hidden: np.ndarray, layer_id: int, *, return_aux: bool = False):
    if not isinstance(hidden, np.ndarray):
        raise TypeError(f"hidden must be a numpy.ndarray, got {type(hidden)}")

    if hidden.ndim != 1:
        raise RuntimeError(f"hidden must be 1-D, got shape={hidden.shape}")

    if hidden.dtype != np.float32:
        raise RuntimeError(f"hidden must have dtype float32, got {hidden.dtype}")

    if not hidden.flags["C_CONTIGUOUS"]:
        hidden = np.ascontiguousarray(hidden)

    router_cfg = get_router_config(session)
    gate_weight, e_score_correction_bias = get_router_tensors(session, layer_id)

    hidden_size = int(router_cfg["hidden_size"])
    if hidden.shape[0] != hidden_size:
        raise RuntimeError(
            f"hidden size mismatch: got={hidden.shape[0]} expected={hidden_size}"
        )

    resident_expert_ids = sorted({int(p["expert_id"]) for p in session.coord.placements})
    if not resident_expert_ids:
        raise RuntimeError("no resident experts found in current placement")

    routes, aux = route_token_real(
        hidden,
        gate_weight,
        e_score_correction_bias,
        n_group=int(router_cfg["n_group"]),
        topk_group=int(router_cfg["topk_group"]),
        top_k=min(int(router_cfg["top_k"]), len(resident_expert_ids)),
        norm_topk_prob=bool(router_cfg["norm_topk_prob"]),
        routed_scaling_factor=float(router_cfg["routed_scaling_factor"]),
        scoring_func=str(router_cfg["scoring_func"]),
        topk_method=str(router_cfg["topk_method"]),
        n_routed_experts=int(router_cfg["n_routed_experts"]),
        hidden_size=int(router_cfg["hidden_size"]),
        resident_expert_ids=resident_expert_ids,
    )

    combined, weighted_outputs = run_topk_moe_layer(session, hidden, routes)

    if return_aux:
        return {
            "output": combined,
            "routes": routes,
            "weighted_outputs": weighted_outputs,
            "aux": aux,
        }

    return {
        "output": combined,
        "routes": routes,
    }
