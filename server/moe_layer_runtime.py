import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from common.protocol import ActivationDType
from server.expert_placement import (
    make_global_expert_id,
    split_global_expert_id,
    allowed_local_expert_ids_for_layer,
    find_expert_placement,
)
from server.expert_placement import split_global_expert_id
from server.fp8_utils import dequant_fp8_weight_blockwise
from server.test_utils import make_safe_input, print_stats, compare_arrays


def infer_one_expert(session, expert_id: int, hidden: np.ndarray):
    expert_id = int(expert_id)

    coord = session.coord
    target = find_expert_placement(coord.placements, expert_id)
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


def route_token_real(
    hidden: np.ndarray,
    gate_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    *,
    n_group: int,
    topk_group: int,
    top_k: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
    scoring_func: str,
    topk_method: str,
    n_routed_experts: int,
    hidden_size: int,
    resident_expert_ids=None,
):
    if scoring_func != "sigmoid":
        raise RuntimeError(f"unsupported scoring_func={scoring_func}")
    if topk_method != "noaux_tc":
        raise RuntimeError(f"unsupported topk_method={topk_method}")

    h = torch.from_numpy(hidden.astype(np.float32))

    if gate_weight.ndim != 2:
        raise RuntimeError(f"gate_weight must be 2D, got shape={tuple(gate_weight.shape)}")
    if e_score_correction_bias.ndim != 1:
        raise RuntimeError(
            f"e_score_correction_bias must be 1D, got shape={tuple(e_score_correction_bias.shape)}"
        )

    num_experts, gate_hidden = gate_weight.shape
    if topk_group <= 0 or topk_group > n_group:
        raise RuntimeError(f"invalid topk_group={topk_group} for n_group={n_group}")
    if top_k <= 0 or top_k > num_experts:
        raise RuntimeError(f"invalid top_k={top_k} for num_experts={num_experts}")
    if num_experts != n_routed_experts:
        raise RuntimeError(
            f"gate_weight shape {tuple(gate_weight.shape)} inconsistent with n_routed_experts={n_routed_experts}"
        )
    if gate_hidden != hidden_size:
        raise RuntimeError(
            f"gate_weight shape {tuple(gate_weight.shape)} inconsistent with hidden_size={hidden_size}"
        )
    if h.shape[0] != hidden_size:
        raise RuntimeError(
            f"hidden dim {h.shape[0]} inconsistent with hidden_size={hidden_size}"
        )
    if e_score_correction_bias.shape[0] != num_experts:
        raise RuntimeError(
            f"bias shape {tuple(e_score_correction_bias.shape)} incompatible with num_experts={num_experts}"
        )
    if num_experts % n_group != 0:
        raise RuntimeError(f"num_experts={num_experts} not divisible by n_group={n_group}")

    experts_per_group = num_experts // n_group

    with torch.no_grad():
        logits = gate_weight @ h
        scores = torch.sigmoid(logits)

    scores_for_choice = scores + e_score_correction_bias

    if resident_expert_ids is None:
        resident_mask = torch.ones(num_experts, dtype=torch.bool)
    else:
        resident_mask = torch.zeros(num_experts, dtype=torch.bool)
        for eid in resident_expert_ids:
            eid = int(eid)
            if 0 <= eid < num_experts:
                resident_mask[eid] = True

        allowed = int(resident_mask.sum().item())
        if allowed == 0:
            raise RuntimeError("resident_expert_ids produced an empty resident mask")

    scores_for_choice = scores_for_choice.masked_fill(~resident_mask, float("-inf"))
    scores = scores.masked_fill(~resident_mask, 0.0)

    grouped = scores_for_choice.view(n_group, experts_per_group)
    top2_per_group = torch.topk(grouped, k=2, dim=-1).values
    group_scores = top2_per_group.sum(dim=-1)

    selected_group_idx = torch.topk(group_scores, k=topk_group, dim=-1).indices

    group_mask = torch.zeros(n_group, dtype=torch.bool)
    group_mask[selected_group_idx] = True

    expert_mask = group_mask.unsqueeze(-1).expand(n_group, experts_per_group).reshape(num_experts)

    masked_scores_for_choice = scores_for_choice.masked_fill(~expert_mask, float("-inf"))

    effective_top_k = min(top_k, int(resident_mask.sum().item()))
    topk_choice_vals, topk_idx = torch.topk(masked_scores_for_choice, k=effective_top_k, dim=-1)

    topk_weight = scores.gather(0, topk_idx)

    if norm_topk_prob and effective_top_k > 1:
        denom = topk_weight.sum() + 1e-20
        topk_weight = topk_weight / denom

    topk_weight = topk_weight * routed_scaling_factor

    routes = [(int(eid), float(w)) for eid, w in zip(topk_idx.tolist(), topk_weight.tolist())]

    aux = {
        "logits": logits.detach().cpu().numpy(),
        "scores": scores.detach().cpu().numpy(),
        "scores_for_choice": scores_for_choice.detach().cpu().numpy(),
        "group_scores": group_scores.detach().cpu().numpy(),
        "selected_group_idx": selected_group_idx.detach().cpu().numpy(),
        "topk_idx": topk_idx.detach().cpu().numpy(),
        "topk_weight": topk_weight.detach().cpu().numpy(),
        "topk_choice_vals": topk_choice_vals.detach().cpu().numpy(),
    }
    return routes, aux


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

    router_cfg = session.get_router_config()
    gate_weight, e_score_correction_bias = (
        session.get_deepseek_model_loader().load_router_tensors_fp32(layer_id)
    )

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
    expert_id = int(expert_id)

    w_up = _load_one_weight_tensor(session, expert_id, "w_up")
    w_gate = _load_one_weight_tensor(session, expert_id, "w_gate")
    w_down = _load_one_weight_tensor(session, expert_id, "w_down")

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
