import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from common.protocol import ActivationDType
from server.expert_id import (
    make_global_expert_id,
    allowed_local_expert_ids_for_layer,
)
from server.expert_id import split_global_expert_id
from server.fp8_utils import dequant_fp8_weight_blockwise
from server.model_locator import resolve_deepseek_tensor_file
from server.router_runtime import (
    get_router_config,
    get_router_tensors,
    route_token_real,
)
from server.test_utils import make_safe_input, print_stats, compare_arrays


def _find_target_placement(coord, expert_id: int):
    expert_id = int(expert_id)
    for p in coord.placements:
        if int(p["expert_id"]) == expert_id:
            return p
    raise RuntimeError(f"expert {expert_id} not found in placements")


def infer_one_expert(session, expert_id: int, hidden: np.ndarray):
    expert_id = int(expert_id)

    coord = session.coord
    target = _find_target_placement(coord, expert_id)
    host = target["host"]
    port = target["worker_port"]

    pool = session.client_pool
    hidden_fp16 = hidden.astype(np.float16, copy=False)

    try:
        client = pool.get(host, port)
        resp = client.send_infer_request(
            {
                "expert_id": expert_id,
                "batch_size": 1,
                "hidden_dim": int(hidden.shape[0]),
                "input_dtype": int(ActivationDType.FP16),
                "output_dtype": int(ActivationDType.FP16),
                "activation": hidden_fp16.tobytes(),
            }
        )
    except Exception:
        pool.invalidate(host, port)
        raise

    if resp["status_code"] != 0:
        raise RuntimeError(
            f"infer failed for expert {expert_id}: "
            f"status={resp['status_code']} output_dtype={resp.get('output_dtype')} "
            f"output_bytes={len(resp['output'])}"
        )

    resp_output_dtype = int(resp["output_dtype"])
    if resp_output_dtype == int(ActivationDType.FP16):
        y = np.frombuffer(resp["output"], dtype=np.float16)
    elif resp_output_dtype == int(ActivationDType.BF16):
        y = np.frombuffer(resp["output"], dtype=ml_dtypes.bfloat16)
    else:
        raise RuntimeError(
            f"unexpected output dtype for expert {expert_id}: {resp_output_dtype}"
        )

    if y.size != hidden.shape[0]:
        raise RuntimeError(
            f"unexpected output size for expert {expert_id}: "
            f"got={y.size} expected={hidden.shape[0]}"
        )

    return {
        "output_dtype": resp_output_dtype,
        "output": resp["output"],
        "array": y,
    }


def _load_one_weight_tensor(model_root: str, layer_id: int, expert_id: int, tensor_kind: str):
    tensor_name, shard_path = resolve_deepseek_tensor_file(
        model_root=model_root,
        layer_id=layer_id,
        expert_id=expert_id,
        tensor_kind=tensor_kind,
    )

    scale_name = tensor_name + "_scale_inv"

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        t = f.get_tensor(tensor_name)

        if t.dtype == torch.float8_e4m3fn:
            keys = set(f.keys())
            if scale_name not in keys:
                raise RuntimeError(f"missing scale tensor for fp8 weight: {scale_name}")

            scale_inv = f.get_tensor(scale_name).to(torch.float32).contiguous()
            t = dequant_fp8_weight_blockwise(t, scale_inv).to(torch.float32).contiguous()

            print(
                f"[top8-ref] loaded {tensor_kind}: "
                f"name={tensor_name} shape={tuple(t.shape)} dtype={t.dtype} "
                f"(dequant from torch.float8_e4m3fn using {scale_name})"
            )
        else:
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
        resp = infer_one_expert(session, expert_id, hidden)
        y = resp["array"]
        weighted_outputs.append((expert_id, weight, y))
        print(
            f"[dispatch] expert={expert_id} weight={weight:.6f} "
            f"finite={np.isfinite(y).sum()}/{y.size} "
            f"min={np.nanmin(y):.6e} max={np.nanmax(y):.6e} "
            f"mean={np.nanmean(y):.6e}"
        )
    return weighted_outputs


def combine_outputs(weighted_outputs):
    if not weighted_outputs:
        raise RuntimeError("weighted_outputs is empty")

    y0 = weighted_outputs[0][2]
    combined = np.zeros_like(y0, dtype=np.float32)

    for expert_id, weight, y in weighted_outputs:
        y = np.asarray(y, dtype=np.float32)
        combined += float(weight) * y

    return combined


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

    layer_id = int(layer_id)

    router_cfg = get_router_config(session)
    gate_weight, e_score_correction_bias = get_router_tensors(session, layer_id)

    hidden_size = int(router_cfg["hidden_size"])
    if hidden.shape[0] != hidden_size:
        raise RuntimeError(
            f"hidden size mismatch: got={hidden.shape[0]} expected={hidden_size}"
        )

    experts_per_layer = int(session.cfg["run"].get("experts_per_layer", 256))

    restricted_local_ids = session.cfg["run"].get("restricted_expert_ids")
    if restricted_local_ids is not None:
        resident_local_expert_ids = sorted({int(x) for x in restricted_local_ids})
        max_local = int(router_cfg["n_routed_experts"])
        for eid in resident_local_expert_ids:
            if eid < 0 or eid >= max_local:
                raise RuntimeError(
                    f"restricted expert id out of range for layer {layer_id}: "
                    f"{eid} not in [0, {max_local})"
                )
    else:
        resident_local_expert_ids = allowed_local_expert_ids_for_layer(
            [int(p["expert_id"]) for p in session.coord.placements],
            layer_id=layer_id,
            experts_per_layer=experts_per_layer,
        )

    if not resident_local_expert_ids:
        raise RuntimeError(f"no resident experts found for layer {layer_id}")

    routes, aux = route_token_real(
        hidden,
        gate_weight,
        e_score_correction_bias,
        n_group=int(router_cfg["n_group"]),
        topk_group=int(router_cfg["topk_group"]),
        top_k=min(int(router_cfg["top_k"]), len(resident_local_expert_ids)),
        norm_topk_prob=bool(router_cfg["norm_topk_prob"]),
        routed_scaling_factor=float(router_cfg["routed_scaling_factor"]),
        scoring_func=str(router_cfg["scoring_func"]),
        topk_method=str(router_cfg["topk_method"]),
        n_routed_experts=int(router_cfg["n_routed_experts"]),
        hidden_size=int(router_cfg["hidden_size"]),
        resident_expert_ids=resident_local_expert_ids,
    )

    global_routes = [
        (
            make_global_expert_id(
                layer_id,
                local_expert_id,
                experts_per_layer=experts_per_layer,
            ),
            weight,
        )
        for local_expert_id, weight in routes
    ]

    combined, weighted_outputs = run_topk_moe_layer(session, hidden, global_routes)

    if return_aux:
        return {
            "output": combined,
            "routes": global_routes,
            "local_routes": routes,
            "weighted_outputs": weighted_outputs,
            "aux": aux,
        }

    return {
        "output": combined,
        "routes": global_routes,
    }


def run_one_expert_reference(session, expert_id: int, hidden: np.ndarray):
    hidden = np.asarray(hidden, dtype=np.float32).reshape(-1)

    model_root = str(session.cfg["model"]["root"])
    experts_per_layer = int(session.cfg["run"].get("experts_per_layer", 256))

    layer_id, local_expert_id = split_global_expert_id(
        int(expert_id),
        experts_per_layer=experts_per_layer,
    )

    w_up = _load_one_weight_tensor(model_root, layer_id, local_expert_id, "w_up")
    w_gate = _load_one_weight_tensor(model_root, layer_id, local_expert_id, "w_gate")
    w_down = _load_one_weight_tensor(model_root, layer_id, local_expert_id, "w_down")

    w_up_np = w_up.numpy()
    w_gate_np = w_gate.numpy()
    w_down_np = w_down.numpy()

    up = w_up_np @ hidden
    gate = w_gate_np @ hidden

    up_t = torch.from_numpy(up.astype(np.float32, copy=False))
    gate_t = torch.from_numpy(gate.astype(np.float32, copy=False))
    fused = (F.silu(gate_t) * up_t).numpy()

    y = w_down_np @ fused
    return y.astype(np.float32, copy=False)


def run_topk_reference(session, routes, hidden: np.ndarray):
    hidden = np.asarray(hidden, dtype=np.float32).reshape(-1)

    weighted_outputs = []
    for expert_id, weight in routes:
        y = run_one_expert_reference(session, expert_id, hidden)
        weighted_outputs.append((int(expert_id), float(weight), y))

    combined = combine_outputs(weighted_outputs)
    return combined, weighted_outputs
