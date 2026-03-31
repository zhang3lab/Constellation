import numpy as np

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


def run_one_expert_correctness_test(session, expert_id: int):
    expert_id = int(expert_id)

    hidden_dim = 7168
    x = make_safe_input(hidden_dim).astype(np.float32, copy=False)

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


def run_multi_expert_correctness_test(session, expert_ids):
    for expert_id in expert_ids:
        print("\n" + "=" * 80)
        print(f"[correctness] testing expert_id={expert_id}")
        run_one_expert_correctness_test(session, expert_id)


def run_one_expert_stability_test(session, expert_id: int, repeats: int = 10):
    expert_id = int(expert_id)

    hidden_dim = 7168
    x = make_safe_input(hidden_dim).astype(np.float32, copy=False)

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
