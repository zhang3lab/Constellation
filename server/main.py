import struct
from common.protocol import TensorKind
from server.client import NodeClient
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

    batch_size = 4
    hidden_dim = 7168
    activation = struct.pack(
        "<" + "f" * (batch_size * hidden_dim),
        *([0.0] * (batch_size * hidden_dim)),
    )

    target = None
    for p in coord.placements:
        if p["expert_id"] == expert_id:
            target = p
            break

    if target is None:
        raise RuntimeError(f"expert {expert_id} not found in placements")

    client = NodeClient(target["host"], target["control_port"])
    with client:
        resp = client.send_infer_request(
            {
                "expert_id": expert_id,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "activation": activation,
            }
        )

    print(
        f"infer response: status={resp['status_code']} "
        f"batch={resp['batch_size']} hidden={resp['hidden_dim']} "
        f"output_bytes={len(resp['output'])}"
    )


if __name__ == "__main__":
    main()
