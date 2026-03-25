import numpy as np
import torch
import torch.nn.functional as F

from server.model_locator import (
    resolve_deepseek_tensor_file,
    resolve_and_load_deepseek_tensor,
)
from server.test_utils import (
    make_safe_input,
    print_stats,
    compare_arrays,
    compare_stability,
)
from safetensors import safe_open


def _infer_once(client, expert_id: int, x: np.ndarray, hidden_dim: int):
    resp = client.send_infer_request(
        {
            "expert_id": expert_id,
            "batch_size": 1,
            "hidden_dim": hidden_dim,
            "activation": x.tobytes(),
        }
    )

    if resp["status_code"] != 0:
        raise RuntimeError(
            f"infer failed: status={resp['status_code']} "
            f"output_bytes={len(resp['output'])}"
        )

    y = np.frombuffer(resp["output"], dtype=np.float16).astype(np.float32)
    if y.size != hidden_dim:
        raise RuntimeError(f"unexpected output size: got {y.size}, expected {hidden_dim}")
    return y


def _find_target_placement(coord, expert_id: int):
    for p in coord.placements:
        if int(p["expert_id"]) == int(expert_id):
            return p
    raise RuntimeError(f"expert {expert_id} not found in placements")


def _infer_with_session_pool(session, expert_id: int, x: np.ndarray, hidden_dim: int):
    target = _find_target_placement(session.coord, expert_id)
    host = target["host"]
    port = target["control_port"]

    pool = session.client_pool

    try:
        client = pool.get(host, port)
        return _infer_once(client, expert_id, x, hidden_dim)
    except Exception:
        pool.invalidate(host, port)
        raise


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
        f"[correctness] loaded {tensor_kind}: "
        f"name={tensor_name} shape={tuple(t.shape)} dtype={t.dtype}"
    )
    return t


def _load_weight_triplet(model_root: str, layer_id: int, expert_id: int):
    w_up = _load_one_weight_tensor(model_root, layer_id, expert_id, "w_up")
    w_gate = _load_one_weight_tensor(model_root, layer_id, expert_id, "w_gate")
    w_down = _load_one_weight_tensor(model_root, layer_id, expert_id, "w_down")
    return w_up, w_gate, w_down


def run_one_expert_correctness_test(session, expert_id: int):
    coord = session.coord
    cfg = session.cfg
    model = cfg["model"]
    run_cfg = cfg["run"]

    layer_id = int(run_cfg["layer_id"])
    model_root = str(model["root"])
    chunk_size = int(model["chunk_size"])

    def tensor_loader(eid: int, tensor_kind_name: str):
        return resolve_and_load_deepseek_tensor(
            model_root=model_root,
            layer_id=layer_id,
            expert_id=eid,
            tensor_kind=tensor_kind_name,
        )

    batch_size = 1
    hidden_dim = 7168
    x = make_safe_input(hidden_dim)

    y_srv = _infer_with_session_pool(session, expert_id, x, hidden_dim)

    print(
        f"[correctness] infer response: "
        f"status=0 batch={batch_size} hidden={hidden_dim} "
        f"output_bytes={y_srv.size * 2}"
    )
    if y_srv.size != hidden_dim:
        raise RuntimeError(f"unexpected output size: got {y_srv.size}, expected {hidden_dim}")

    W_up, W_gate, W_down = _load_weight_triplet(
        model_root=model_root,
        layer_id=layer_id,
        expert_id=expert_id,
    )

    x_t = torch.from_numpy(x)

    print(f"[correctness] W_up shape   = {tuple(W_up.shape)}")
    print(f"[correctness] W_gate shape = {tuple(W_gate.shape)}")
    print(f"[correctness] W_down shape = {tuple(W_down.shape)}")

    print_stats("x", x_t)
    print_stats("W_up", W_up)
    print_stats("W_gate", W_gate)
    print_stats("W_down", W_down)

    up = W_up @ x_t
    gate = W_gate @ x_t
    fused = up * F.silu(gate)
    y_ref = W_down @ fused

    print_stats("up", up)
    print_stats("gate", gate)
    print_stats("fused", fused)
    print_stats("y_ref_fp32", y_ref)

    y_ref_cmp = y_ref.to(torch.float16).to(torch.float32).cpu().numpy()

    print_stats("y_ref_fp16_roundtrip", y_ref_cmp)
    print_stats("y_srv", y_srv)

    print("[correctness] x[:8]      =", x[:8])
    print("[correctness] y_ref[:8]  =", y_ref_cmp[:8])
    print("[correctness] y_srv[:8]  =", y_srv[:8])

    compare_arrays("output", y_ref_cmp, y_srv)

def run_multi_expert_correctness_test(session, expert_ids):
    for expert_id in expert_ids:
        print("\n" + "=" * 80)
        print(f"[correctness] testing expert_id={expert_id}")
        run_one_expert_correctness_test(session, expert_id)


def run_one_expert_stability_test(session, expert_id: int, repeats: int = 10):
    coord = session.coord
    cfg = session.cfg
    model = cfg["model"]
    run_cfg = cfg["run"]

    layer_id = int(run_cfg["layer_id"])
    model_root = str(model["root"])
    chunk_size = int(model["chunk_size"])

    def tensor_loader(eid: int, tensor_kind_name: str):
        return resolve_and_load_deepseek_tensor(
            model_root=model_root,
            layer_id=layer_id,
            expert_id=eid,
            tensor_kind=tensor_kind_name,
        )

    hidden_dim = 7168
    x = make_safe_input(hidden_dim)

    outputs = []
    for i in range(repeats):
        y = _infer_with_session_pool(session, expert_id, x, hidden_dim)
        outputs.append(y)
        print(
            f"[stability] expert={expert_id} iter={i} "
            f"min={y.min():.6e} max={y.max():.6e} mean={y.mean():.6e}"
        )

    ref = outputs[0]
    for i in range(1, repeats):
        compare_stability(f"expert={expert_id} run0_vs_run{i}", ref, outputs[i])
