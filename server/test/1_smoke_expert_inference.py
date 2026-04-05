import argparse
from pathlib import Path

import numpy as np
import torch

from server.moe_layer_runtime import (
    infer_one_expert,
    run_one_expert_reference,
)
from server.test.utils import (
    make_safe_input,
    print_stats,
    compare_arrays,
    compare_stability,
)


def load_input_vector(
    *,
    input_pt: str | None,
    token_idx: int | None,
    hidden_dim: int,
) -> np.ndarray:
    if input_pt is None:
        return make_safe_input(hidden_dim).astype(np.float32, copy=False)

    x = torch.load(input_pt, map_location="cpu")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor in {input_pt}, got {type(x).__name__}")

    x = x.detach().float().cpu()

    if x.ndim == 1:
        vec = x
    elif x.ndim == 2:
        if token_idx is None:
            raise RuntimeError(
                f"{input_pt} has shape {tuple(x.shape)}; please provide --token-idx"
            )
        if token_idx < 0 or token_idx >= x.shape[0]:
            raise RuntimeError(
                f"token_idx out of range: {token_idx} for shape {tuple(x.shape)}"
            )
        vec = x[token_idx]
    elif x.ndim == 3:
        if x.shape[0] != 1:
            raise RuntimeError(
                f"unsupported 3D input shape {tuple(x.shape)}; expected batch size 1"
            )
        if token_idx is None:
            raise RuntimeError(
                f"{input_pt} has shape {tuple(x.shape)}; please provide --token-idx"
            )
        if token_idx < 0 or token_idx >= x.shape[1]:
            raise RuntimeError(
                f"token_idx out of range: {token_idx} for shape {tuple(x.shape)}"
            )
        vec = x[0, token_idx]
    else:
        raise RuntimeError(f"unsupported input tensor shape: {tuple(x.shape)}")

    vec = np.asarray(vec.numpy(), dtype=np.float32)
    if vec.ndim != 1:
        raise RuntimeError(f"expected 1D vector after selection, got shape {vec.shape}")
    if vec.shape[0] != hidden_dim:
        raise RuntimeError(
            f"hidden dim mismatch: got {vec.shape[0]}, expected {hidden_dim}"
        )
    return vec


def run_one_expert_correctness_test(
    session,
    expert_id: int,
    *,
    input_pt: str | None = None,
    token_idx: int | None = None,
):
    expert_id = int(expert_id)

    hidden_dim = 7168
    x = load_input_vector(
        input_pt=input_pt,
        token_idx=token_idx,
        hidden_dim=hidden_dim,
    ).astype(np.float32, copy=False)

    infer_resp = infer_one_expert(session, expert_id, x)
    y_srv = np.asarray(infer_resp["array"], dtype=np.float32)

    print(
        f"[correctness] infer response: "
        f"status=0 hidden={hidden_dim} "
        f"output_dtype={infer_resp['output_dtype']} "
        f"output_bytes={len(infer_resp['output'])}"
    )

    if y_srv.size != hidden_dim:
        raise RuntimeError(
            f"unexpected output size for expert {expert_id}: "
            f"got {y_srv.size}, expected {hidden_dim}"
        )

    y_ref = run_one_expert_reference(session, expert_id, x)
    y_ref = np.asarray(y_ref, dtype=np.float32)

    print_stats("x", x)
    print_stats("y_ref", y_ref)
    print_stats("y_srv", y_srv)

    print("[correctness] x[:8]     =", x[:8])
    print("[correctness] y_ref[:8] =", y_ref[:8])
    print("[correctness] y_srv[:8] =", y_srv[:8])

    compare_arrays("output", y_ref, y_srv)


def run_multi_expert_correctness_test(
    session,
    expert_ids,
    *,
    input_pt: str | None = None,
    token_idx: int | None = None,
):
    for expert_id in expert_ids:
        print("\n" + "=" * 80)
        print(f"[correctness] testing expert_id={expert_id}")
        run_one_expert_correctness_test(
            session,
            expert_id,
            input_pt=input_pt,
            token_idx=token_idx,
        )


def run_one_expert_stability_test(
    session,
    expert_id: int,
    *,
    repeats: int = 10,
    input_pt: str | None = None,
    token_idx: int | None = None,
):
    expert_id = int(expert_id)

    hidden_dim = 7168
    x = load_input_vector(
        input_pt=input_pt,
        token_idx=token_idx,
        hidden_dim=hidden_dim,
    ).astype(np.float32, copy=False)

    outputs = []
    for i in range(repeats):
        infer_resp = infer_one_expert(session, expert_id, x)
        y = np.asarray(infer_resp["array"], dtype=np.float32)

        outputs.append(y)
        print(
            f"[stability] expert={expert_id} iter={i} "
            f"output_dtype={infer_resp['output_dtype']} "
            f"min={y.min():.6e} max={y.max():.6e} mean={y.mean():.6e}"
        )

    ref = outputs[0]
    for i in range(1, repeats):
        compare_stability(f"expert={expert_id} run0_vs_run{i}", ref, outputs[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--expert-id", type=int, required=True)
    ap.add_argument("--input-pt", type=str, default=None)
    ap.add_argument("--token-idx", type=int, default=None)
    ap.add_argument("--mode", type=str, default="correctness", choices=["correctness", "stability"])
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    from server.config import load_config
    from server.control_plane import setup_control_plane
    from server.coordinator import Coordinator
    from server.inference_session import InferenceSession

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    with InferenceSession(coord, cfg) as session:
        if args.mode == "correctness":
            run_one_expert_correctness_test(
                session,
                args.expert_id,
                input_pt=args.input_pt,
                token_idx=args.token_idx,
            )
        else:
            run_one_expert_stability_test(
                session,
                args.expert_id,
                repeats=args.repeats,
                input_pt=args.input_pt,
                token_idx=args.token_idx,
            )


if __name__ == "__main__":
    main()
