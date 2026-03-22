from server.config import load_config
from server.coordinator import Coordinator
from server.model_locator import resolve_deepseek_tensor_file


def main():
    cfg = load_config("server/config.json")

    coord = Coordinator(cfg["nodes"])
    coord.discover_nodes()
    coord.print_summary()

    coord.build_placement(
        num_experts=8,
        expert_mem_bytes=2 * 1024**3,
        memory_utilization=0.9,
    )
    coord.print_placement()

    coord.send_placement_plan()

    test_load = cfg["test_load"]
    model = cfg["model"]

    tensor_name, shard_path = resolve_deepseek_tensor_file(
        model_root=model["root"],
        layer_id=int(test_load["layer_id"]),
        expert_id=int(test_load["expert_id"]),
        tensor_kind=str(test_load["tensor_kind"]),
    )

    print(f"resolved tensor_name={tensor_name}")
    print(f"resolved shard_path={shard_path}")


if __name__ == "__main__":
    main()
