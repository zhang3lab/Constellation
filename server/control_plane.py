from server.model_locator import (
    resolve_and_load_deepseek_tensor,
    resolve_and_load_deepseek_scale_tensor,
)
from server.router_runtime import load_router_config


def build_tensor_loader(cfg):
    model = cfg["model"]
    run_cfg = cfg["run"]

    model_root = str(model["root"])
    layer_id = int(run_cfg["layer_id"])

    def tensor_loader(eid: int, tensor_kind_name: str):
        if tensor_kind_name in ("w_up", "w_gate", "w_down"):
            return resolve_and_load_deepseek_tensor(
                model_root=model_root,
                layer_id=layer_id,
                expert_id=eid,
                tensor_kind=tensor_kind_name,
            )

        if tensor_kind_name == "w_up_scale":
            return resolve_and_load_deepseek_scale_tensor(
                model_root=model_root,
                layer_id=layer_id,
                expert_id=eid,
                tensor_kind="w_up",
            )

        if tensor_kind_name == "w_gate_scale":
            return resolve_and_load_deepseek_scale_tensor(
                model_root=model_root,
                layer_id=layer_id,
                expert_id=eid,
                tensor_kind="w_gate",
            )

        if tensor_kind_name == "w_down_scale":
            return resolve_and_load_deepseek_scale_tensor(
                model_root=model_root,
                layer_id=layer_id,
                expert_id=eid,
                tensor_kind="w_down",
            )

        raise RuntimeError(f"unsupported tensor_kind_name: {tensor_kind_name}")

    return tensor_loader


def setup_control_plane(coord, cfg):
    model = cfg["model"]
    run_cfg = cfg["run"]

    model_root = str(model["root"])
    chunk_size = int(model["chunk_size"])
    expert_mem_bytes = int(model["expert_mem_bytes"])
    memory_utilization = float(model["memory_utilization"])
    num_experts = int(run_cfg["num_experts"])

    coord.discover_nodes()
    coord.print_summary()

    coord.build_placement(
        num_experts=num_experts,
        expert_mem_bytes=expert_mem_bytes,
        memory_utilization=memory_utilization,
    )
    coord.print_placement()

    coord.send_placement_plan()

    tensor_loader = build_tensor_loader(cfg)
    coord.preload_all_placed_experts(
        tensor_loader=tensor_loader,
        chunk_size=chunk_size,
    )
