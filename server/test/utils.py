import numpy as np
import torch


def make_safe_input(hidden_dim: int):
    x = np.load("server/debug_hidden_layer3.npy").astype(np.float32, copy=False).reshape(-1)
    if x.shape[0] != hidden_dim:
        raise RuntimeError(
            f"debug hidden size mismatch: got={x.shape[0]} expected={hidden_dim}"
        )
    return x


def stats_str(name, x):
    if isinstance(x, torch.Tensor):
        a = x.detach()
        if a.dtype != torch.float32:
            a = a.to(torch.float32)
        finite = torch.isfinite(a)
        finite_count = int(finite.sum().item())
        total = a.numel()

        if finite_count == 0:
            return f"[stats] {name}: shape={tuple(a.shape)} finite=0/{total} all non-finite"

        af = a[finite]
        return (
            f"[stats] {name}: shape={tuple(a.shape)} finite={finite_count}/{total} "
            f"min={float(af.min().item()):.6e} max={float(af.max().item()):.6e} "
            f"mean={float(af.mean().item()):.6e} std={float(af.std(unbiased=False).item()):.6e}"
        )

    a = to_numpy_f32(x)
    finite = np.isfinite(a)
    finite_count = int(finite.sum())
    total = a.size

    if finite_count == 0:
        return f"[stats] {name}: shape={a.shape} finite=0/{total} all non-finite"

    af = a[finite]
    return (
        f"[stats] {name}: shape={a.shape} finite={finite_count}/{total} "
        f"min={af.min():.6e} max={af.max():.6e} "
        f"mean={af.mean():.6e} std={af.std():.6e}"
    )


def print_stats(name, x):
    print(stats_str(name, x))


def compare_arrays(name, ref, got):
    ref = to_numpy_f32(ref).reshape(-1)
    got = to_numpy_f32(got).reshape(-1)

    diff = np.abs(ref - got)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    denom = np.maximum(np.abs(ref), 1e-8)
    max_rel = float((diff / denom).max())
    cos = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))

    print(
        f"[compare] {name}: "
        f"max_abs={max_abs:.6e} "
        f"mean_abs={mean_abs:.6e} "
        f"max_rel={max_rel:.6e} "
        f"cos={cos:.8f}"
    )


def compare_stability(name, a, b):
    a = to_numpy_f32(a).reshape(-1)
    b = to_numpy_f32(b).reshape(-1)

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


def to_numpy_f32(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)

def prenorm_hidden_for_attention(
    session,
    hidden_in,
    layer_id: int,
):
    if session.backbone_store is None:
        raise RuntimeError("session.backbone_store is not initialized")

    layer_entry = session.backbone_store.layer(int(layer_id))
    norm_weight = layer_entry["input_layernorm"]

    if not isinstance(norm_weight, torch.Tensor):
        raise TypeError(
            f"layer{layer_id}.input_layernorm expected torch.Tensor, got {type(norm_weight).__name__}"
        )

    hidden_t = torch.as_tensor(hidden_in, dtype=torch.float32, device=norm_weight.device)
    if hidden_t.ndim == 1:
        hidden_t = hidden_t.unsqueeze(0)

    norm_weight = norm_weight.to(device=hidden_t.device, dtype=hidden_t.dtype)
    hidden_norm = torch.nn.functional.rms_norm(
        hidden_t,
        (hidden_t.shape[-1],),
        norm_weight,
        1e-6,
    )

    if hidden_norm.shape[0] == 1:
        hidden_norm = hidden_norm[0]

    return hidden_norm.detach().cpu().numpy().astype(np.float32, copy=False)
