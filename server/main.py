from common.protocol import TensorKind
from server.config import load_config
from server.coordinator import Coordinator


def parse_tensor_kind(name: str) -> TensorKind:
    table = {
        "w_up": TensorKind.WUp,
        "w_gate": TensorKind.WGate,
        "w_down": TensorKind.WDown,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"unknown tensor_kind: {name}") from exc


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

    coord.send_one_tensor_file(
        expert_id=int(test_load["expert_id"]),
        tensor_kind=parse_tensor_kind(test_load["tensor_kind"]),
        path=model["root"],   # 这里后面会改成“根目录 + index 查 tensor”
        chunk_size=int(model["chunk_size"]),
    )


if __name__ == "__main__":
    main()
