import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from server.test_utils import make_safe_input, compare_stability


def load_router_config(model_root: str):
    config_path = Path(model_root) / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"config.json not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    router_cfg = {
        "n_group": int(cfg["n_group"]),
        "topk_group": int(cfg["topk_group"]),
        "top_k": int(cfg["num_experts_per_tok"]),
        "norm_topk_prob": bool(cfg["norm_topk_prob"]),
        "routed_scaling_factor": float(cfg["routed_scaling_factor"]),
        "scoring_func": str(cfg["scoring_func"]),
        "topk_method": str(cfg["topk_method"]),
        "n_routed_experts": int(cfg["n_routed_experts"]),
        "hidden_size": int(cfg["hidden_size"]),
    }

    print(f"[router] loaded router config from {config_path}: {router_cfg}")
    return router_cfg


def load_router_tensors(model_root: str, layer_id: int):
    tensor_weight = f"model.layers.{layer_id}.mlp.gate.weight"
    tensor_bias = f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias"

    shard_paths = sorted(Path(model_root).rglob("*.safetensors"))
    if not shard_paths:
        raise RuntimeError(f"no safetensors found under {model_root}")

    found_weight = None
    found_bias = None

    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())

            if found_weight is None and tensor_weight in keys:
                found_weight = f.get_tensor(tensor_weight).to(torch.float32).contiguous()
                print(
                    f"[router] loaded weight: {tensor_weight} "
                    f"from {shard_path} shape={tuple(found_weight.shape)} dtype={found_weight.dtype}"
                )

            if found_bias is None and tensor_bias in keys:
                found_bias = f.get_tensor(tensor_bias).to(torch.float32).contiguous()
                print(
                    f"[router] loaded bias: {tensor_bias} "
                    f"from {shard_path} shape={tuple(found_bias.shape)} dtype={found_bias.dtype}"
                )

            if found_weight is not None and found_bias is not None:
                break

    if found_weight is None:
        raise RuntimeError(f"router weight not found: {tensor_weight}")
    if found_bias is None:
        raise RuntimeError(f"router bias not found: {tensor_bias}")

    return found_weight, found_bias


def get_router_config(session):
    if session.router_cfg is None:
        model_root = str(session.cfg["model"]["root"])
        session.router_cfg = load_router_config(model_root)
    return session.router_cfg


def get_router_tensors(session, layer_id: int):
    layer_id = int(layer_id)
    cached = session.router_tensors_by_layer.get(layer_id)
    if cached is not None:
        return cached

    model_root = str(session.cfg["model"]["root"])
    gate_weight, e_score_correction_bias = load_router_tensors(model_root, layer_id)

    session.router_tensors_by_layer[layer_id] = (
        gate_weight,
        e_score_correction_bias,
    )
    return session.router_tensors_by_layer[layer_id]


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
