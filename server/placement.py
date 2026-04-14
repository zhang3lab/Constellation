from typing import Any, Dict, List

from common.protocol import GpuVendor


class PlacementError(RuntimeError):
    pass


def build_balanced_placement(
    gpu_inventory: List[Dict[str, Any]],
    expert_ids: List[int],
    expert_mem_bytes: int,
    memory_utilization: float = 0.9,
) -> List[Dict[str, Any]]:
    if expert_mem_bytes <= 0:
        raise ValueError(f"expert_mem_bytes must be > 0, got {expert_mem_bytes}")
    if not (0.0 < memory_utilization <= 1.0):
        raise ValueError(
            f"memory_utilization must be in (0, 1], got {memory_utilization}"
        )

    gpus = []
    for gpu in gpu_inventory:
        if gpu.get("gpu_vendor") == GpuVendor.CPU_FP16_RESIDENT:
            continue

        capacity_bytes = int(gpu["free_mem_bytes"] * memory_utilization)
        g = dict(gpu)
        g["capacity_bytes"] = capacity_bytes
        g["remaining_mem_bytes"] = capacity_bytes
        g["assigned_slot_ids"] = []
        g["resident_expert_ids"] = set(gpu.get("resident_expert_ids", []))
        gpus.append(g)

    if not gpus:
        raise PlacementError("no eligible workers found for placement")

    placements: List[Dict[str, Any]] = []

    for placement_index, expert_id in enumerate(expert_ids):
        reusable = []
        fresh = []

        for gpu in gpus:
            if expert_id in gpu["resident_expert_ids"]:
                assigned_count = len(gpu["assigned_slot_ids"])
                key = (0, assigned_count)
                reusable.append((key, gpu))
            elif gpu["remaining_mem_bytes"] >= expert_mem_bytes:
                after_bytes = gpu["remaining_mem_bytes"] - expert_mem_bytes
                util_after = 1.0 - (after_bytes / max(gpu["capacity_bytes"], 1))
                assigned_count = len(gpu["assigned_slot_ids"])
                key = (1, assigned_count, util_after)
                fresh.append((key, gpu))

        chosen = None

        if reusable:
            reusable.sort(key=lambda x: x[0])
            chosen = reusable[0][1]
        elif fresh:
            fresh.sort(key=lambda x: x[0])
            chosen = fresh[0][1]
            chosen["remaining_mem_bytes"] -= expert_mem_bytes
        else:
            max_remaining = max((gpu["remaining_mem_bytes"] for gpu in gpus), default=0)
            raise PlacementError(
                f"unable to place slot {placement_index} expert={expert_id}: "
                f"need {expert_mem_bytes} bytes, "
                f"max remaining across eligible workers is {max_remaining} bytes"
            )

        chosen["assigned_slot_ids"].append(expert_id)

        placements.append(
            {
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
                "reuse_existing_resident": expert_id in chosen["resident_expert_ids"],
            }
        )

    return placements
