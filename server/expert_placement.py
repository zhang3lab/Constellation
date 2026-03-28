from typing import Iterable


def make_global_expert_id(
    layer_id: int,
    local_expert_id: int,
    experts_per_layer: int = 256,
) -> int:
    layer_id = int(layer_id)
    local_expert_id = int(local_expert_id)
    experts_per_layer = int(experts_per_layer)

    if layer_id < 0:
        raise ValueError(f"layer_id must be >= 0, got {layer_id}")
    if local_expert_id < 0:
        raise ValueError(f"local_expert_id must be >= 0, got {local_expert_id}")
    if experts_per_layer <= 0:
        raise ValueError(f"experts_per_layer must be > 0, got {experts_per_layer}")
    if local_expert_id >= experts_per_layer:
        raise ValueError(
            f"local_expert_id must be < experts_per_layer, got "
            f"{local_expert_id} vs {experts_per_layer}"
        )

    return layer_id * experts_per_layer + local_expert_id


def split_global_expert_id(
    global_expert_id: int,
    experts_per_layer: int = 256,
) -> tuple[int, int]:
    global_expert_id = int(global_expert_id)
    experts_per_layer = int(experts_per_layer)

    if global_expert_id < 0:
        raise ValueError(f"global_expert_id must be >= 0, got {global_expert_id}")
    if experts_per_layer <= 0:
        raise ValueError(f"experts_per_layer must be > 0, got {experts_per_layer}")

    return global_expert_id // experts_per_layer, global_expert_id % experts_per_layer


def allowed_local_expert_ids_for_layer(
    global_expert_ids: Iterable[int],
    layer_id: int,
    experts_per_layer: int = 256,
) -> list[int]:
    layer_id = int(layer_id)
    out = []

    for gid in global_expert_ids:
        gid_layer, gid_local = split_global_expert_id(
            int(gid),
            experts_per_layer=experts_per_layer,
        )
        if gid_layer == layer_id:
            out.append(gid_local)

    return sorted(set(out))


def find_expert_placement(placements, expert_id: int):
    expert_id = int(expert_id)
    for p in placements:
        if int(p["expert_id"]) == expert_id:
            return p
    raise RuntimeError(f"expert {expert_id} not found in placements")


def group_placements_by_control_endpoint(placements):
    groups = {}
    for p in placements:
        key = (str(p["host"]), int(p["control_port"]))
        groups.setdefault(key, []).append(p)

    for key in groups:
        groups[key].sort(key=lambda p: int(p["expert_id"]))

    return groups
