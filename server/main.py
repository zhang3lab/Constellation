from common.protocol import TensorKind
from server.config import load_config
from server.coordinator import Coordinator
from server.model_locator import resolve_and_load_deepseek_tensor


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

    model = cfg["model"]
    test_load = cfg["test_load"]

    tensor_name, shard_path, tensor_bytes, shape, dtype = resolve_and_load_deepseek_tensor(
        model_root=model["root"],
        layer_id=int(test_load["layer_id"]),
        expert_id=int(test_load["expert_id"]),
        tensor_kind=str(test_load["tensor_kind"]),
    )

    print(f"resolved tensor_name={tensor_name}")
    print(f"resolved shard_path={shard_path}")
    print(f"resolved shape={shape} dtype={dtype} total_bytes={len(tensor_bytes)}")

    coord.send_one_tensor_bytes(
        expert_id=int(test_load["expert_id"]),
        tensor_kind=parse_tensor_kind(str(test_load["tensor_kind"])),
        tensor_bytes=tensor_bytes,
        chunk_size=int(model["chunk_size"]),
    )


if __name__ == "__main__":
    main()
