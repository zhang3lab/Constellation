import numpy as np
import torch
import torch.nn.functional as F

from server.client import NodeClient
from server.model_locator import resolve_and_load_deepseek_tensor


def _compare(name, ref, got):
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    got = np.asarray(got, dtype=np.float32).reshape(-1)

    if ref.shape != got.shape:
        raise RuntimeError(f"{name}: shape mismatch: ref={ref.shape}, got={got.shape}")

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


def _find_target_placement(coord, expert_id: int):
    for p in coord.placements:
        if int(p["expert_id"]) == int(expert_id):
            return p
    raise RuntimeError(f"expert {expert_id} not found in placements")


def _load_weight_triplet(model_root: str, layer_id: int, expert_id: int):
    w_up = resolve_and_load_deepseek_tensor(
        model_root=model_root,
        layer_id=layer_id,
        expert_id=expert_id,
        tensor_kind="w_up",
    )
    w_gate = resolve_and_load_deepseek_tensor(
        model_root=model_root,
        layer_id=layer_id,
        expert_id=expert_id,
        tensor_kind="w_gate",
    )
    w_down = resolve_and_load_deepseek_tensor(
        model_root=model_root,
        layer_id=layer_id,
        expert_id=expert_id,
        tensor_kind="w_down",
    )
    return w_up, w_gate, w_down


def _to_torch_f32(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float32).contiguous()
    return torch.as_tensor(x, dtype=torch.float32).contiguous()


def run_single_expert_correctness_test(coord, cfg):
    model = cfg["model"]
    test_load = cfg["test_load"]

    layer_id = int(test_load["layer_id"])
    expert_id = int(test_load["expert_id"])
    model_root = str(model["root"])

    batch_size = 1
    hidden_dim = 7168

    rng = np.random.default_rng(0)
    x = (rng.standard_normal(hidden_dim).astype(np.float32) * 0.1)

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
        raise RuntimeError(
            f"unexpected output size: got {y_srv.size}, expected {hidden_dim}"
        )

    w_up, w_gate, w_down = _load_weight_triplet(
        model_root=model_root,
        layer_id=layer_id,
        expert_id=expert_id,
    )

    W_up = _to_torch_f32(w_up)
    W_gate = _to_torch_f32(w_gate)
    W_down = _to_torch_f32(w_down)
    x_t = torch.from_numpy(x)

    print(f"[correctness] W_up shape   = {tuple(W_up.shape)}")
    print(f"[correctness] W_gate shape = {tuple(W_gate.shape)}")
    print(f"[correctness] W_down shape = {tuple(W_down.shape)}")

    # kernel 语义：
    # up   = W_up @ x
    # gate = W_gate @ x
    # y    = W_down @ (up * silu(gate))
    up = W_up @ x_t
    gate = W_gate @ x_t
    fused = up * F.silu(gate)
    y_ref = W_down @ fused

    # server 最终输出是 fp16，所以 reference 先做 roundtrip 再比较
    y_ref_cmp = y_ref.to(torch.float16).to(torch.float32).cpu().numpy()

    print("[correctness] x[:8]      =", x[:8])
    print("[correctness] y_ref[:8]  =", y_ref_cmp[:8])
    print("[correctness] y_srv[:8]  =", y_srv[:8])

    _compare("output", y_ref_cmp, y_srv)
