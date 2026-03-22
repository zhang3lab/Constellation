from common.protocol import TensorKind
from server.config import load_config
from server.coordinator import Coordinator
from server.model_locator import resolve_and_load_deepseek_tensor


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

    model = cfg["model"]
    test_load = cfg["test_load"]

    layer_id = int(test_load["layer_id"])
    expert_id = int(test_load["expert_id"])
    chunk_size = int(model["chunk_size"])
    model_root = str(model["root"])

    def tensor_loader(eid: int, tensor_kind_name: str):
        return resolve_and_load_deepseek_tensor(
            model_root=model_root,
            layer_id=layer_id,
            expert_id=eid,
            tensor_kind=tensor_kind_name,
        )

    coord.send_one_expert_triplet(
        expert_id=expert_id,
        tensor_loader=tensor_loader,
        chunk_size=chunk_size,
    )


if __name__ == "__main__":
    main()
