import argparse

from server.config import load_config
from server.coordinator import Coordinator
from server.control_plane import build_restricted_global_expert_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])

    model = cfg["model"]
    run_cfg = cfg["run"]

    expert_mem_bytes = int(model["expert_mem_bytes"])
    memory_utilization = float(model["memory_utilization"])

    preload_expert_ids = build_restricted_global_expert_ids(run_cfg)
    if preload_expert_ids is not None:
        preload_expert_ids = [int(x) for x in preload_expert_ids]
        if len(set(preload_expert_ids)) != len(preload_expert_ids):
            raise RuntimeError("preload_expert_ids contains duplicates")
        num_experts = len(preload_expert_ids)
    else:
        num_experts = int(run_cfg["num_experts"])

    print("[placement-e2e] discover before placement")
    coord.discover_nodes()
    coord.print_summary()

    if not coord.node_inventories:
        raise RuntimeError("no node inventories discovered")
    if not coord.gpu_inventory:
        raise RuntimeError("no gpu inventory discovered")

    coord.build_placement(
        num_experts=num_experts,
        expert_mem_bytes=expert_mem_bytes,
        memory_utilization=memory_utilization,
    )

    if preload_expert_ids is not None:
        if len(coord.placements) != len(preload_expert_ids):
            raise RuntimeError(
                f"placement size mismatch: placements={len(coord.placements)} "
                f"restricted={len(preload_expert_ids)}"
            )
        for p, eid in zip(coord.placements, preload_expert_ids):
            p["expert_id"] = int(eid)

    if not coord.placements:
        raise RuntimeError("placement is empty")

    print(f"[placement-e2e] built placements={len(coord.placements)}")
    coord.print_placement()

    drop_non_target_residents = bool(run_cfg.get("drop_non_target_residents", False))
    placement_acks = coord.send_placement_plan(
        drop_non_target_residents=drop_non_target_residents
    )

    if len(placement_acks) != len(coord.node_inventories):
        raise RuntimeError(
            f"placement ack size mismatch: acks={len(placement_acks)} "
            f"nodes={len(coord.node_inventories)}"
        )

    total_target = 0
    total_ready = 0
    needs_load_nodes = 0

    ack_by_node = {}
    for ack in placement_acks:
        node_instance_id = str(ack["node_instance_id"])
        ack_by_node[node_instance_id] = ack

        status_code = int(ack["status_code"])
        if status_code != 0:
            raise RuntimeError(
                f"placement ack error for node={node_instance_id}: status_code={status_code}"
            )

        num_sent = int(ack["num_assignments_sent"])
        num_target = int(ack["num_target_experts"])
        num_ready = int(ack["num_ready_experts"])
        needs_load = bool(ack["needs_load"])
        all_ready = bool(ack["all_ready"])

        if num_target != num_sent:
            raise RuntimeError(
                f"target/sent mismatch for node={node_instance_id}: "
                f"sent={num_sent} target={num_target}"
            )

        if num_ready < 0 or num_ready > num_target:
            raise RuntimeError(
                f"bad ready count for node={node_instance_id}: "
                f"ready={num_ready} target={num_target}"
            )

        if all_ready != (num_ready == num_target):
            raise RuntimeError(
                f"all_ready mismatch for node={node_instance_id}: "
                f"all_ready={all_ready} ready={num_ready} target={num_target}"
            )

        if needs_load != (not all_ready):
            raise RuntimeError(
                f"needs_load mismatch for node={node_instance_id}: "
                f"needs_load={needs_load} all_ready={all_ready}"
            )

        total_target += num_target
        total_ready += num_ready
        if needs_load:
            needs_load_nodes += 1

        print(
            f"[placement-e2e] ack "
            f"node={node_instance_id} "
            f"sent={num_sent} target={num_target} ready={num_ready} "
            f"needs_load={int(needs_load)} all_ready={int(all_ready)} "
            f"drop_non_target_residents={int(bool(ack['drop_non_target_residents']))}"
        )

    if total_target != len(coord.placements):
        raise RuntimeError(
            f"global target mismatch: total_target={total_target} "
            f"placements={len(coord.placements)}"
        )

    print(
        f"[placement-e2e] totals placements={len(coord.placements)} "
        f"ready={total_ready} needs_load_nodes={needs_load_nodes}"
    )

    print("[placement-e2e] discover after placement")
    coord.discover_nodes()

    placement_keys = {
        (
            str(p["node_instance_id"]),
            int(p["worker_id"]),
            int(p["expert_id"]),
        )
        for p in coord.placements
    }

    resident_hits = 0
    for node_instance_id, resident_by_worker in coord.node_resident_inventories.items():
        for worker_id, expert_ids in resident_by_worker.items():
            for expert_id in expert_ids:
                key = (str(node_instance_id), int(worker_id), int(expert_id))
                if key in placement_keys:
                    resident_hits += 1

    print(f"[placement-e2e] resident_hits_after_placement={resident_hits}")

    # This test validates the control-plane placement path itself.
    # We do not require resident hits here because placement alone may not load weights.
    print("PASS=1")


if __name__ == "__main__":
    main()
