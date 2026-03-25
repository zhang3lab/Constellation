from server.config import load_config
from server.coordinator import Coordinator
from server.inference_session import InferenceSession
from server.model_locator import resolve_and_load_deepseek_tensor
from server.moe_layer_runtime import run_moe_layer
from server.router_runtime import load_router_config, run_one_token_moe_real_router
from server.test_utils import make_safe_input
from server.validation_suite import run_validation_suite


def build_tensor_loader(cfg):
    model = cfg["model"]
    run_cfg = cfg["run"]

    model_root = str(model["root"])
    layer_id = int(run_cfg["layer_id"])

    def tensor_loader(eid: int, tensor_kind_name: str):
        return resolve_and_load_deepseek_tensor(
            model_root=model_root,
            layer_id=layer_id,
            expert_id=eid,
            tensor_kind=tensor_kind_name,
        )

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


def run_runtime_validation(coord, cfg):
    with InferenceSession(coord, cfg) as session:
        run_validation_suite(session)


def run_runtime_demo(coord, cfg):
    run_cfg = cfg["run"]
    layer_id = int(run_cfg["layer_id"])
    model_root = str(cfg["model"]["root"])

    router_cfg = load_router_config(model_root)
    hidden_size = int(router_cfg["hidden_size"])

    with InferenceSession(coord, cfg) as session:
        hidden = make_safe_input(hidden_size)
        result = run_moe_layer(session, hidden, layer_id)

        print("[demo] routes =", result["routes"])
        print("[demo] combined[:8] =", result["combined"][:8])


def main():
    cfg = load_config("server/config.json")

    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    mode = cfg["run"].get("mode", "validation")
    if mode == "validation":
        run_runtime_validation(coord, cfg)
    elif mode == "demo":
        run_runtime_demo(coord, cfg)
    else:
        raise RuntimeError(f"unknown mode={mode}")


if __name__ == "__main__":
    main()
