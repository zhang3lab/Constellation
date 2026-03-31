from server.expert_placement import make_global_expert_id, split_global_expert_id
from server.deepseek_model_loader import DeepseekModelLoader


def build_restricted_global_expert_ids(run_cfg):
    restricted_local_ids = run_cfg.get("restricted_expert_ids")
    if restricted_local_ids is None:
        return None

    restricted_local_ids = [int(x) for x in restricted_local_ids]
    if not restricted_local_ids:
        raise RuntimeError("restricted_expert_ids is provided but empty")

    experts_per_layer = int(run_cfg["experts_per_layer"])
    sparse_layer_start = int(run_cfg["sparse_layer_start"])
    sparse_layer_end = int(run_cfg["sparse_layer_end"])

    if experts_per_layer <= 0:
        raise RuntimeError(
            f"experts_per_layer must be > 0, got {experts_per_layer}"
        )
    if sparse_layer_end < sparse_layer_start:
        raise RuntimeError(
            f"invalid sparse layer range: start={sparse_layer_start} end={sparse_layer_end}"
        )

    for eid in restricted_local_ids:
        if eid < 0 or eid >= experts_per_layer:
            raise RuntimeError(
                f"restricted expert id out of range: {eid} not in [0, {experts_per_layer})"
            )

    out = []
    for layer_id in range(sparse_layer_start, sparse_layer_end + 1):
        for local_expert_id in restricted_local_ids:
            out.append(
                make_global_expert_id(
                    layer_id,
                    local_expert_id,
                    experts_per_layer=experts_per_layer,
                )
            )
    return out


def setup_control_plane(coord, cfg):
    model = cfg["model"]
    run_cfg = cfg["run"]

    model_root = str(model["root"])
    chunk_size = int(model["chunk_size"])
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

    coord.discover_nodes()
    coord.print_summary()

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

    if cfg["verbose"]:
        coord.print_placement()
    placement_acks = coord.send_placement_plan()

    model_loader = DeepseekModelLoader(model_root)
    coord.preload_all_placed_experts(
        model_loader=model_loader,
        chunk_size=chunk_size,
        experts_per_layer=int(run_cfg["experts_per_layer"]),
        placement_acks=placement_acks,
    )
