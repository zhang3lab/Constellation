from server.expert_inference_validation import (
    run_multi_expert_correctness_test,
    run_one_expert_stability_test,
)
from server.moe_layer_runtime import run_top8_reference_compare_test
from server.router_runtime import run_real_router_demo


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
