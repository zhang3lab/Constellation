import numpy as np

from server.expert_placement import make_global_expert_id
from server.expert_inference_validation import (
    run_multi_expert_correctness_test,
    run_one_expert_stability_test,
)
from server.moe_layer_runtime import (
    run_topk_moe_layer,
    run_topk_reference,
    run_moe_layer,
)
from server.test_utils import make_safe_input, print_stats, compare_arrays, compare_stability


def _get_experts_per_layer(session) -> int:
    return int(session.cfg["run"]["experts_per_layer"])


def _local_to_global(layer_id: int, local_expert_id: int, experts_per_layer: int) -> int:
    return make_global_expert_id(
        layer_id=layer_id,
        local_expert_id=local_expert_id,
        experts_per_layer=experts_per_layer,
    )


def _make_global_expert_ids_for_layer(session, layer_id: int, local_expert_ids):
    experts_per_layer = _get_experts_per_layer(session)
    return [
        _local_to_global(layer_id, int(local_eid), experts_per_layer)
        for local_eid in local_expert_ids
    ]


def run_real_router_demo(session, layer_id: int, repeats: int = 10):
    hidden_size = int(session.get_router_config()["hidden_size"])
    hidden = make_safe_input(hidden_size)

    result = run_moe_layer(session, hidden, layer_id, return_aux=True)

    print("[router] global routes =", result["routes"])
    print("[router] local routes  =", result["local_routes"])
    print("[router] selected_group_idx =", result["aux"]["selected_group_idx"])
    print("[router] topk_idx =", result["aux"]["topk_idx"])
    print("[router] topk_weight =", result["aux"]["topk_weight"])
    print("[router] topk_weight_sum =", float(np.sum(result["aux"]["topk_weight"])))
    print("[router] output[:8] =", result["output"][:8])

    out = np.asarray(result["output"], dtype=np.float32)
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
    hidden_size = int(session.get_router_config()["hidden_size"])
    hidden = make_safe_input(hidden_size)

    outputs = []
    routes_ref = None
    local_routes_ref = None
    topk_idx_ref = None
    topk_weight_ref = None

    for i in range(repeats):
        result = run_moe_layer(session, hidden, layer_id, return_aux=True)
        routes = result["routes"]
        local_routes = result["local_routes"]
        out = np.asarray(result["output"], dtype=np.float32)
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
            local_routes_ref = local_routes
            topk_idx_ref = np.asarray(aux["topk_idx"])
            topk_weight_ref = np.asarray(aux["topk_weight"], dtype=np.float32)
        else:
            if routes != routes_ref:
                raise RuntimeError(
                    f"router global routes changed between runs:\n"
                    f"ref={routes_ref}\n"
                    f"cur={routes}"
                )

            if local_routes != local_routes_ref:
                raise RuntimeError(
                    f"router local routes changed between runs:\n"
                    f"ref={local_routes_ref}\n"
                    f"cur={local_routes}"
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


def run_top8_reference_compare_test(session, layer_id: int):
    hidden_size = int(session.get_router_config()["hidden_size"])
    x = make_safe_input(hidden_size)

    global_expert_ids = _make_global_expert_ids_for_layer(
        session,
        layer_id=layer_id,
        local_expert_ids=range(8),
    )
    routes = [(eid, 1.0 / 8.0) for eid in global_expert_ids]
    print(f"[top8] global routes={routes}")

    combined_srv, outputs_srv = run_topk_moe_layer(session, x, routes)
    combined_ref, outputs_ref = run_topk_reference(session, routes, x)

    combined_srv = np.asarray(combined_srv, dtype=np.float32)
    combined_ref = np.asarray(combined_ref, dtype=np.float32)

    print_stats("combined_srv", combined_srv)
    print_stats("combined_ref", combined_ref)

    print("[top8] combined_srv[:8] =", combined_srv[:8])
    print("[top8] combined_ref[:8] =", combined_ref[:8])

    for (eid_s, w_s, y_s), (eid_r, w_r, y_r) in zip(outputs_srv, outputs_ref):
        if eid_s != eid_r:
            raise RuntimeError(f"expert order mismatch: runtime={eid_s}, ref={eid_r}")
        if abs(float(w_s) - float(w_r)) > 1e-12:
            raise RuntimeError(f"weight mismatch for expert {eid_s}: runtime={w_s}, ref={w_r}")

        y_s = np.asarray(y_s, dtype=np.float32)
        y_r = np.asarray(y_r, dtype=np.float32)

        print_stats(f"expert{eid_s}_srv", y_s)
        print_stats(f"expert{eid_s}_ref", y_r)
        print(f"[top8] expert={eid_s} weight={w_s:.6f}")
        print(f"[top8] expert{eid_s}_srv[:4] =", y_s[:4])
        print(f"[top8] expert{eid_s}_ref[:4] =", y_r[:4])
        compare_arrays(f"top8 expert{eid_s}_srv_vs_ref", y_r, y_s)

    compare_arrays("top8 combined_srv_vs_ref", combined_ref, combined_srv)


def run_validation_suite(session):
    layer_id = int(session.cfg["run"]["layer_id"])

    correctness_expert_ids = _make_global_expert_ids_for_layer(
        session,
        layer_id=layer_id,
        local_expert_ids=[0, 1, 2],
    )
    stability_expert_ids = _make_global_expert_ids_for_layer(
        session,
        layer_id=layer_id,
        local_expert_ids=[0, 1],
    )

    print("\n" + "=" * 80)
    print("[suite] multi-expert correctness")
    run_multi_expert_correctness_test(session, expert_ids=correctness_expert_ids)

    print("\n" + "=" * 80)
    print("[suite] expert stability")
    for expert_id in stability_expert_ids:
        run_one_expert_stability_test(session, expert_id=expert_id, repeats=10)

    print("\n" + "=" * 80)
    print("[suite] top8 reference compare")
    run_top8_reference_compare_test(session, layer_id=layer_id)

    print("\n" + "=" * 80)
    print("[suite] real router demo + stability")
    run_real_router_demo(session, layer_id=layer_id, repeats=10)

    print("\n" + "=" * 80)
    print("ALL VALIDATION PASSED")
