from typing import Any, Dict, List

from common.protocol import GpuVendor


class PlacementError(RuntimeError):
    pass


def build_balanced_placement(
    gpu_inventory: List[Dict[str, Any]],
    expert_ids: List[int],
    expert_mem_bytes: int,
    memory_utilization: float = 0.9,
    allow_drop_non_target_residents: bool = False,
):
    if expert_mem_bytes <= 0:
        raise ValueError(f"expert_mem_bytes must be > 0, got {expert_mem_bytes}")
    if not (0.0 < memory_utilization <= 1.0):
        raise ValueError(
            f"memory_utilization must be in (0, 1], got {memory_utilization}"
        )
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError("expert_ids contains duplicates")

    gpus = []
    drop_non_target_residents_by_node: Dict[str, bool] = {}

    for gpu in gpu_inventory:
        if gpu.get("gpu_vendor") == GpuVendor.CPU_FP16_RESIDENT:
            continue

        node_instance_id = str(gpu["node_instance_id"])
        capacity_bytes = int(gpu["total_mem_bytes"] * memory_utilization)
        used_mem_bytes = int(gpu["total_mem_bytes"] - gpu["free_mem_bytes"])
        remaining_mem_bytes = max(0, capacity_bytes - used_mem_bytes)

        g = dict(gpu)
        g["capacity_bytes"] = capacity_bytes
        g["remaining_mem_bytes"] = remaining_mem_bytes
        g["assigned_expert_ids"] = set()
        g["resident_expert_ids"] = set(gpu.get("resident_expert_ids", []))
        g["droppable_resident_count"] = 0
        gpus.append(g)

        if node_instance_id not in drop_non_target_residents_by_node:
            drop_non_target_residents_by_node[node_instance_id] = False

    if not gpus:
        raise PlacementError("no eligible workers found for placement")

    def placement_row(chosen: Dict[str, Any], expert_id: int, reused: bool) -> Dict[str, Any]:
        return {
            "expert_id": expert_id,
            "node_instance_id": chosen["node_instance_id"],
            "reported_node_id": chosen["reported_node_id"],
            "host": chosen["host"],
            "control_port": chosen["control_port"],
            "gpu_uid_global": chosen["gpu_uid_global"],
            "gpu_uid_reported": chosen["gpu_uid_reported"],
            "worker_id": chosen["worker_id"],
            "worker_port": chosen["worker_port"],
            "gpu_name": chosen["gpu_name"],
            "expert_mem_bytes": expert_mem_bytes,
            "reuse_existing_resident": reused,
        }

    def apply_node_drop(node_instance_id: str) -> None:
        if drop_non_target_residents_by_node[node_instance_id]:
            return
        for gpu in gpus:
            if str(gpu["node_instance_id"]) != node_instance_id:
                continue
            gpu["remaining_mem_bytes"] += (
                gpu["droppable_resident_count"] * expert_mem_bytes
            )
            gpu["droppable_resident_count"] = 0
        drop_non_target_residents_by_node[node_instance_id] = True

    placements: List[Dict[str, Any]] = []
    placed_expert_ids = set()

    # Round 0: reuse existing resident experts whenever possible.
    for expert_id in expert_ids:
        reusable = []

        for gpu in gpus:
            if expert_id not in gpu["resident_expert_ids"]:
                continue
            assigned_count = len(gpu["assigned_expert_ids"])
            after_bytes = gpu["remaining_mem_bytes"]
            util_after = 1.0 - (after_bytes / max(gpu["capacity_bytes"], 1))
            key = (assigned_count, util_after)
            reusable.append((key, gpu))

        if not reusable:
            continue

        reusable.sort(key=lambda x: x[0])
        chosen = reusable[0][1]
        chosen["assigned_expert_ids"].add(expert_id)
        placed_expert_ids.add(expert_id)
        placements.append(placement_row(chosen, expert_id, reused=True))

    # Compute droppable residents only after reuse decisions are fixed.
    for gpu in gpus:
        gpu["droppable_resident_count"] = len(
            gpu["resident_expert_ids"] - gpu["assigned_expert_ids"]
        )

    fresh_expert_ids = [
        expert_id for expert_id in expert_ids
        if expert_id not in placed_expert_ids
    ]

    # Round 1: fresh placement. Prefer direct fit; only trigger node drop when needed.
    for fresh_index, expert_id in enumerate(fresh_expert_ids):
        direct_candidates = []

        for gpu in gpus:
            if gpu["remaining_mem_bytes"] < expert_mem_bytes:
                continue
            assigned_count = len(gpu["assigned_expert_ids"])
            after_bytes = gpu["remaining_mem_bytes"] - expert_mem_bytes
            util_after = 1.0 - (after_bytes / max(gpu["capacity_bytes"], 1))
            key = (assigned_count, util_after)
            direct_candidates.append((key, gpu))

        if direct_candidates:
            direct_candidates.sort(key=lambda x: x[0])
            chosen = direct_candidates[0][1]
            chosen["remaining_mem_bytes"] -= expert_mem_bytes
            chosen["assigned_expert_ids"].add(expert_id)
            placements.append(placement_row(chosen, expert_id, reused=False))
            continue

        drop_candidates = []

        if allow_drop_non_target_residents:
            for node_instance_id, already_drop in drop_non_target_residents_by_node.items():
                if already_drop:
                    continue

                node_gpus = [
                    gpu for gpu in gpus
                    if str(gpu["node_instance_id"]) == node_instance_id
                ]
                node_droppable_count = sum(
                    gpu["droppable_resident_count"] for gpu in node_gpus
                )
                if node_droppable_count <= 0:
                    continue

                best_key = None
                for gpu in node_gpus:
                    effective_remaining = (
                        gpu["remaining_mem_bytes"] +
                        gpu["droppable_resident_count"] * expert_mem_bytes
                    )
                    if effective_remaining < expert_mem_bytes:
                        continue

                    assigned_count = len(gpu["assigned_expert_ids"])
                    after_bytes = max(gpu["remaining_mem_bytes"] - expert_mem_bytes, 0)
                    util_after = 1.0 - (after_bytes / max(gpu["capacity_bytes"], 1))
                    key = (assigned_count, util_after)

                    if best_key is None or key < best_key:
                        best_key = key

                if best_key is not None:
                    drop_candidates.append((best_key, node_instance_id))

        if drop_candidates:
            drop_candidates.sort(key=lambda x: x[0])
            node_to_drop = drop_candidates[0][1]
            apply_node_drop(node_to_drop)

            direct_candidates = []
            for gpu in gpus:
                if str(gpu["node_instance_id"]) != node_to_drop:
                    continue
                if gpu["remaining_mem_bytes"] < expert_mem_bytes:
                    continue
                assigned_count = len(gpu["assigned_expert_ids"])
                after_bytes = gpu["remaining_mem_bytes"] - expert_mem_bytes
                util_after = 1.0 - (after_bytes / max(gpu["capacity_bytes"], 1))
                key = (assigned_count, util_after)
                direct_candidates.append((key, gpu))

            if not direct_candidates:
                raise PlacementError(
                    f"internal placement error after applying drop on "
                    f"node={node_to_drop} for expert={expert_id}"
                )

            direct_candidates.sort(key=lambda x: x[0])
            chosen = direct_candidates[0][1]
            chosen["remaining_mem_bytes"] -= expert_mem_bytes
            chosen["assigned_expert_ids"].add(expert_id)
            placements.append(placement_row(chosen, expert_id, reused=False))
            continue

        max_remaining = max((gpu["remaining_mem_bytes"] for gpu in gpus), default=0)
        print(
            f"[placement] FAIL fresh_index={fresh_index} expert={expert_id} "
            f"need={expert_mem_bytes} max_remaining={max_remaining}"
        )
        for gpu in sorted(gpus, key=lambda x: x["remaining_mem_bytes"], reverse=True):
            print(
                f"[placement] worker-state "
                f"node={gpu['node_instance_id']} "
                f"worker={gpu['worker_id']} "
                f"remaining={gpu['remaining_mem_bytes']} "
                f"droppable_count={gpu['droppable_resident_count']} "
                f"assigned={len(gpu['assigned_expert_ids'])} "
                f"resident_count={len(gpu['resident_expert_ids'])} "
                f"node_drop={int(drop_non_target_residents_by_node[str(gpu['node_instance_id'])])}"
            )
        raise PlacementError(
            f"unable to place fresh expert={expert_id}: need {expert_mem_bytes} bytes"
        )

    assert len(placements) == len(expert_ids)
    assert len({p['expert_id'] for p in placements}) == len(expert_ids)

    return placements, drop_non_target_residents_by_node
