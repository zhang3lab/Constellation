import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from server.model_locator import resolve_deepseek_tensor_file
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


def run_one_expert_reference(session, expert_id: int, hidden: np.ndarray):
    cfg = session.cfg
    model = cfg["model"]
    run_cfg = cfg["run"]

    model_root = str(model["root"])
    layer_id = int(run_cfg["layer_id"])

    W_up = _load_one_weight_tensor(model_root, layer_id, expert_id, "w_up")
    W_gate = _load_one_weight_tensor(model_root, layer_id, expert_id, "w_gate")
    W_down = _load_one_weight_tensor(model_root, layer_id, expert_id, "w_down")

    x_t = torch.from_numpy(hidden.astype(np.float32))

    up = W_up @ x_t
    gate = W_gate @ x_t
    fused = up * F.silu(gate)
    y_ref = W_down @ fused

    # 对齐 server 最终 fp16 输出路径
    y_ref = y_ref.to(torch.float16).to(torch.float32).cpu().numpy()
    return y_ref


def combine_outputs(weighted_outputs):
    if not weighted_outputs:
        raise RuntimeError("weighted_outputs is empty")

    hidden_dim = weighted_outputs[0][2].shape[0]
    combined = np.zeros(hidden_dim, dtype=np.float32)

    for _, weight, y in weighted_outputs:
        combined += float(weight) * y
    return combined


def route_token_top8_uniform():
    return [(eid, 1.0 / 8.0) for eid in range(8)]


def run_top8_reference(session, routes, hidden: np.ndarray):
    weighted_outputs = []
    for expert_id, weight in routes:
        y = run_one_expert_reference(session, expert_id, hidden)
        weighted_outputs.append((expert_id, weight, y))
    combined = combine_outputs(weighted_outputs)
    return combined, weighted_outputs


def run_top8_reference_compare_test(session):
    hidden_dim = 7168
    x = make_safe_input(hidden_dim)

    routes = route_token_top8_uniform()
    print(f"[top8] routes={routes}")

    combined_srv, outputs_srv = run_topk_moe_layer(session, x, routes)
    combined_ref, outputs_ref = run_top8_reference(session, routes, x)

    print_stats("combined_srv", combined_srv)
    print_stats("combined_ref", combined_ref)

    print("[top8] combined_srv[:8] =", combined_srv[:8])
    print("[top8] combined_ref[:8] =", combined_ref[:8])

    for (eid_s, w_s, y_s), (eid_r, w_r, y_r) in zip(outputs_srv, outputs_ref):
        if eid_s != eid_r:
            raise RuntimeError(f"expert order mismatch: runtime={eid_s}, ref={eid_r}")
        if abs(float(w_s) - float(w_r)) > 1e-12:
            raise RuntimeError(f"weight mismatch for expert {eid_s}: runtime={w_s}, ref={w_r}")

        print_stats(f"expert{eid_s}_srv", y_s)
        print_stats(f"expert{eid_s}_ref", y_r)
        print(f"[top8] expert={eid_s} weight={w_s:.6f}")
        print(f"[top8] expert{eid_s}_srv[:4] =", y_s[:4])
        print(f"[top8] expert{eid_s}_ref[:4] =", y_r[:4])
        compare_arrays(f"expert{eid_s}_srv_vs_ref", y_r, y_s)

    compare_arrays("combined_srv_vs_ref", combined_ref, combined_srv)
