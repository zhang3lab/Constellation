import numpy as np

from server.expert_inference_validation import (
    run_multi_expert_correctness_test,
    run_one_expert_stability_test,
)
from server.moe_layer_runtime import run_moe_layer
from server.router_runtime import get_router_config
from server.test_utils import make_safe_input, compare_stability


def run_real_router_demo(session, layer_id: int, repeats: int = 10):
    hidden_size = int(get_router_config(session)["hidden_size"])
    hidden = make_safe_input(hidden_size)

    result = run_moe_layer(session, hidden, layer_id, return_aux=True)

    print("[router] routes =", result["routes"])
    print("[router] selected_group_idx =", result["aux"]["selected_group_idx"])
    print("[router] topk_idx =", result["aux"]["topk_idx"])
    print("[router] topk_weight =", result["aux"]["topk_weight"])
    print("[router] topk_weight_sum =", float(np.sum(result["aux"]["topk_weight"])))
    print("[router] output[:8] =", result["output"][:8])

    out = result["output"]
    finite = np.isfinite(out)
    finite_count = int(finite.sum())
    total = out.size
    print(f"[router] output finite={finite_count}/{total}")
    print(
        f"[router] output stats: "
        f"min={out.min():.6e} max={out.max():.6e} mean={out.mean():.6e}"
    )

    if finite_count != total:
        raise RuntimeError("real-router output contains non-finite values")

    run_real_router_stability_test(session, layer_id=layer_id, repeats=repeats)


def run_real_router_stability_test(session, layer_id: int, repeats: int = 10):
    hidden_size = int(get_router_config(session)["hidden_size"])
    hidden = make_safe_input(hidden_size)

    outputs = []
    routes_ref = None
    topk_idx_ref = None
    topk_weight_ref = None

    for i in range(repeats):
        result = run_moe_layer(session, hidden, layer_id, return_aux=True)
        routes = result["routes"]
        out = result["output"]
        aux = result["aux"]

        print(
            f"[router-stability] iter={i} "
            f"topk_idx={aux['topk_idx']} "
            f"topk_weight={aux['topk_weight']}"
        )
        print(
            f"[router-stability] iter={i} "
            f"min={out.min():.6e} max={out.max():.6e} mean={out.mean():.6e}"
        )

        if routes_ref is None:
            routes_ref = routes
            topk_idx_ref = np.asarray(aux["topk_idx"])
            topk_weight_ref = np.asarray(aux["topk_weight"], dtype=np.float32)
        else:
            if routes != routes_ref:
                raise RuntimeError(
                    f"router routes changed between runs:\n"
                    f"ref={routes_ref}\n"
                    f"cur={routes}"
                )

            if not np.array_equal(np.asarray(aux["topk_idx"]), topk_idx_ref):
                raise RuntimeError(
                    f"router topk_idx changed between runs:\n"
                    f"ref={topk_idx_ref}\n"
                    f"cur={aux['topk_idx']}"
                )

            cur_topk_weight = np.asarray(aux["topk_weight"], dtype=np.float32)
            if not np.array_equal(cur_topk_weight, topk_weight_ref):
                raise RuntimeError(
                    f"router topk_weight changed between runs:\n"
                    f"ref={topk_weight_ref}\n"
                    f"cur={cur_topk_weight}"
                )

        finite = np.isfinite(out)
        if int(finite.sum()) != out.size:
            raise RuntimeError("real-router output contains non-finite values")

        outputs.append(out)

    ref = outputs[0]
    for i in range(1, repeats):
        compare_stability(f"router run0_vs_run{i}", ref, outputs[i])


def run_validation_suite(session):
    layer_id = int(session.cfg["run"]["layer_id"])

    print("\n" + "=" * 80)
    print("[suite] multi-expert correctness")
    run_multi_expert_correctness_test(session, expert_ids=[0, 1, 2])

    print("\n" + "=" * 80)
    print("[suite] expert stability")
    run_one_expert_stability_test(session, expert_id=0, repeats=10)
    run_one_expert_stability_test(session, expert_id=1, repeats=10)

    print("\n" + "=" * 80)
    print("[suite] top8 reference compare")
    run_top8_reference_compare_test(session)

    print("\n" + "=" * 80)
    print("[suite] real router demo + stability")
    run_real_router_demo(session, layer_id=layer_id, repeats=10)
