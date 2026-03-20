from server.config import load_nodes_config
from server.coordinator import Coordinator


def main():
    nodes = load_nodes_config("server/nodes.json")

    coord = Coordinator(nodes)
    coord.discover_nodes()
    coord.print_summary()

    num_experts = 8
    expert_mem_bytes = 2 * 1024**3  # 2 GiB

    coord.build_placement(
        num_experts=num_experts,
        expert_mem_bytes=expert_mem_bytes,
        memory_utilization=0.9,
    )
    coord.print_placement()
    coord.send_placement_plan()
    coord.test_send_load_weights_begin()


if __name__ == "__main__":
    main()
