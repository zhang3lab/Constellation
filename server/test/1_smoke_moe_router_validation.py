from __future__ import annotations

import argparse
import numpy as np

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.inference_session import InferenceSession
from server.moe_layer_runtime import run_moe_layer
from server.test.utils import compare_stability


def run_real_router_demo(session, layer_id: int, repeats: int = 10):
    hidden_size = int(session.get_router_config()["hidden_size"])
    hidden = np.zeros((hidden_size,), dtype=np.float32)

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
    hidden = np.zeros((hidden_size,), dtype=np.float32)

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


def run_moe_router_validation(session, layer_id: int):
    layer_id = int(layer_id)

    print("\n" + "=" * 80)
    print("[suite] real router demo + stability")
    run_real_router_demo(session, layer_id=layer_id, repeats=10)

    print("\n" + "=" * 80)
    print("ALL MOE ROUTER VALIDATION PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--layer-id", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    layer_id = (
        int(args.layer_id)
        if args.layer_id is not None
        else int(cfg["run"]["sparse_layer_start"])
    )

    with InferenceSession(coord, cfg) as session:
        run_moe_router_validation(session, layer_id=layer_id)


if __name__ == "__main__":
    main()
