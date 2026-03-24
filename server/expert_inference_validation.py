import numpy as np
import torch
import torch.nn.functional as F

from server.model_locator import (
    resolve_deepseek_tensor_file,
    resolve_and_load_deepseek_tensor,
)
from safetensors import safe_open


def _compare(name, ref, got):
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    got = np.asarray(got, dtype=np.float32).reshape(-1)

    diff = np.abs(ref - got)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    denom = np.maximum(np.abs(ref), 1e-8)
    max_rel = float((diff / denom).max())
    cos = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))

    print(
        f"[correctness] {name}: "
        f"max_abs={max_abs:.6e} "
        f"mean_abs={mean_abs:.6e} "
        f"max_rel={max_rel:.6e} "
        f"cos={cos:.8f}"
    )


def _compare_two_outputs(name, a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    diff = np.abs(a - b)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    exact = bool(np.array_equal(a, b))
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print(
        f"[stability] {name}: "
        f"exact={exact} "
        f"max_abs={max_abs:.6e} "
        f"mean_abs={mean_abs:.6e} "
        f"cos={cos:.8f}"
    )


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


def _stats(name, t):
    if isinstance(t, torch.Tensor):
        a = t.detach().cpu().float()
        finite = torch.isfinite(a)
        finite_count = int(finite.sum().item())
        total = a.numel()
        print(f"[correctness] {name}: shape={tuple(a.shape)} finite={finite_count}/{total}")
        if finite_count > 0:
            af = a[finite]
            print(
                f"[correctness] {name}: "
                f"min={af.min().item():.6e} "
                f"max={af.max().item():.6e} "
                f"mean={af.mean().item():.6e} "
                f"std={af.std().item():.6e}"
            )
    else:
        a = np.asarray(t, dtype=np.float32)
        finite = np.isfinite(a)
        finite_count = int(finite.sum())
        total = a.size
        print(f"[correctness] {name}: shape={a.shape} finite={finite_count}/{total}")
        if finite_count > 0:
            af = a[finite]
            print(
                f"[correctness] {name}: "
                f"min={af.min():.6e} "
                f"max={af.max():.6e} "
                f"mean={af.mean():.6e} "
                f"std={af.std():.6e}"
            )


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


def _make_safe_test_input(hidden_dim: int):
    x = np.zeros(hidden_dim, dtype=np.float32)
    x[0] = 1e-4
    x[7] = -2e-4
    x[19] = 5e-5
    x[123] = -1e-4
    return x


def run_one_expert_correctness_test(session, expert_id: int):
    coord = session.coord
    cfg = session.cfg
    model = cfg["model"]
    test_load = cfg["test_load"]

    layer_id = int(test_load["layer_id"])
    model_root = str(model["root"])
    chunk_size = int(model["chunk_size"])

    def tensor_loader(eid: int, tensor_kind_name: str):
        return resolve_and_load_deepseek_tensor(
            model_root=model_root,
            layer_id=layer_id,
            expert_id=eid,
            tensor_kind=tensor_kind_name,
        )

    coord.send_one_expert_triplet(
        expert_id=expert_id,
        tensor_loader=tensor_loader,
        chunk_size=chunk_size,
    )

    batch_size = 1
    hidden_dim = 7168
    x = _make_safe_test_input(hidden_dim)

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

    _stats("x", x_t)
    _stats("W_up", W_up)
    _stats("W_gate", W_gate)
    _stats("W_down", W_down)

    up = W_up @ x_t
    gate = W_gate @ x_t
    fused = up * F.silu(gate)
    y_ref = W_down @ fused

    _stats("up", up)
    _stats("gate", gate)
    _stats("fused", fused)
    _stats("y_ref_fp32", y_ref)

    y_ref_cmp = y_ref.to(torch.float16).to(torch.float32).cpu().numpy()

    _stats("y_ref_fp16_roundtrip", y_ref_cmp)
    _stats("y_srv", y_srv)

    print("[correctness] x[:8]      =", x[:8])
    print("[correctness] y_ref[:8]  =", y_ref_cmp[:8])
    print("[correctness] y_srv[:8]  =", y_srv[:8])

    _compare("output", y_ref_cmp, y_srv)

def run_multi_expert_correctness_test(session, expert_ids):
    for expert_id in expert_ids:
        print("\n" + "=" * 80)
        print(f"[correctness] testing expert_id={expert_id}")
        run_one_expert_correctness_test(session, expert_id)


def run_one_expert_stability_test(session, expert_id: int, repeats: int = 10):
    coord = session.coord
    cfg = session.cfg
    model = cfg["model"]
    test_load = cfg["test_load"]

    layer_id = int(test_load["layer_id"])
    model_root = str(model["root"])
    chunk_size = int(model["chunk_size"])

    def tensor_loader(eid: int, tensor_kind_name: str):
        return resolve_and_load_deepseek_tensor(
            model_root=model_root,
            layer_id=layer_id,
            expert_id=eid,
            tensor_kind=tensor_kind_name,
        )

    coord.send_one_expert_triplet(
        expert_id=expert_id,
        tensor_loader=tensor_loader,
        chunk_size=chunk_size,
    )

    hidden_dim = 7168
    x = _make_safe_test_input(hidden_dim)

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
        _compare_two_outputs(f"expert={expert_id} run0_vs_run{i}", ref, outputs[i])
