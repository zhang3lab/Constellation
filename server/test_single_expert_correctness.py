import numpy as np
import torch
import torch.nn.functional as F

from server.client import NodeClient
from server.model_locator import resolve_deepseek_tensor_file
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


def run_one_expert_correctness_test(coord, cfg, expert_id: int):
    model = cfg["model"]
    test_load = cfg["test_load"]

    layer_id = int(test_load["layer_id"])
    model_root = str(model["root"])
    chunk_size = int(model["chunk_size"])

    def tensor_loader(eid: int, tensor_kind_name: str):
        tensor_name, shard_path = resolve_deepseek_tensor_file(
            model_root=model_root,
            layer_id=layer_id,
            expert_id=eid,
            tensor_kind=tensor_kind_name,
        )
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            t = f.get_tensor(tensor_name)
        return tensor_name, shard_path, t.numpy().tobytes(), tuple(t.shape), str(t.dtype)

    coord.send_one_expert_triplet(
        expert_id=expert_id,
        tensor_loader=tensor_loader,
        chunk_size=chunk_size,
    )

    batch_size = 1
    hidden_dim = 7168
    x = _make_safe_test_input(hidden_dim)

    target = _find_target_placement(coord, expert_id)

    client = NodeClient(target["host"], target["control_port"])
    with client:
        resp = client.send_infer_request(
            {
                "expert_id": expert_id,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "activation": x.tobytes(),
            }
        )

    print(
        f"[correctness] infer response: "
        f"status={resp['status_code']} "
        f"batch={resp['batch_size']} "
        f"hidden={resp['hidden_dim']} "
        f"output_bytes={len(resp['output'])}"
    )

    y_srv = np.frombuffer(resp["output"], dtype=np.float16).astype(np.float32)
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

def run_multi_expert_correctness_test(coord, cfg, expert_ids):
    for expert_id in expert_ids:
        print("\n" + "=" * 80)
        print(f"[correctness] testing expert_id={expert_id}")
        run_one_expert_correctness_test(coord, cfg, expert_id)
