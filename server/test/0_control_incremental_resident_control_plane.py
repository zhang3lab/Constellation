import argparse
import struct
from typing import Dict, List, Set, Tuple

from common.protocol import TensorKind
from server.config import load_config
from server.coordinator import Coordinator, NodeClient


def _pack_f32_list(xs: List[float]) -> bytes:
    return b"".join(struct.pack("<f", float(x)) for x in xs)


def _make_dummy_expert_tensors() -> List[dict]:
    # This test only validates real control-plane + weight-upload plumbing.
    # The tensors here are intentionally tiny synthetic payloads rather than
    # real DeepSeek expert weights, so the test stays fast and deterministic.
    #
    # The shapes only need to satisfy the backend structural constraints:
    #   w_up.rows == w_gate.rows
    #   w_up.cols == w_gate.cols
    #   w_down.rows == w_up.cols
    #   w_down.cols == w_up.rows
    up_rows = 4
    up_cols = 8
    down_rows = up_cols
    down_cols = up_rows

    row_block = 1
    col_block = 1

    w_up = bytes((i % 256 for i in range(up_rows * up_cols)))
    w_gate = bytes(((17 + i) % 256 for i in range(up_rows * up_cols)))
    w_down = bytes(((33 + i) % 256 for i in range(down_rows * down_cols)))

    w_up_scale = _pack_f32_list([1.0] * (up_rows * up_cols))
    w_gate_scale = _pack_f32_list([1.0] * (up_rows * up_cols))
    w_down_scale = _pack_f32_list([1.0] * (down_rows * down_cols))

    return [
        {
            "tensor_kind": TensorKind.WUp,
            "tensor_bytes": w_up,
            "shape": [up_rows, up_cols],
            "dtype": "float8_e4m3fn",
            "row_block": row_block,
            "col_block": col_block,
        },
        {
            "tensor_kind": TensorKind.WUpScale,
            "tensor_bytes": w_up_scale,
            "shape": [up_rows, up_cols],
            "dtype": "float32",
            "row_block": row_block,
            "col_block": col_block,
        },
        {
            "tensor_kind": TensorKind.WGate,
            "tensor_bytes": w_gate,
            "shape": [up_rows, up_cols],
            "dtype": "float8_e4m3fn",
            "row_block": row_block,
            "col_block": col_block,
        },
        {
            "tensor_kind": TensorKind.WGateScale,
            "tensor_bytes": w_gate_scale,
            "shape": [up_rows, up_cols],
            "dtype": "float32",
            "row_block": row_block,
            "col_block": col_block,
        },
        {
            "tensor_kind": TensorKind.WDown,
            "tensor_bytes": w_down,
            "shape": [down_rows, down_cols],
            "dtype": "float8_e4m3fn",
            "row_block": row_block,
            "col_block": col_block,
        },
        {
            "tensor_kind": TensorKind.WDownScale,
            "tensor_bytes": w_down_scale,
            "shape": [down_rows, down_cols],
            "dtype": "float32",
            "row_block": row_block,
            "col_block": col_block,
        },
    ]


def _upload_dummy_expert(
    target: dict,
    expert_id: int,
    chunk_size: int,
    log_level: int = 0,
) -> None:
    tensors = _make_dummy_expert_tensors()

    client = NodeClient(
        str(target["host"]),
        int(target["control_port"]),
        log_level=log_level,
    )
    with client:
        for t in tensors:
            begin_msg = {
                "expert_id": int(expert_id),
                "worker_id": int(target["worker_id"]),
                "tensor_kind": t["tensor_kind"],
                "total_bytes": len(t["tensor_bytes"]),
                "meta": {
                    "shape": [int(x) for x in t["shape"]],
                    "dtype": str(t["dtype"]),
                    "row_block": int(t["row_block"]),
                    "col_block": int(t["col_block"]),
                },
            }
            client.send_load_weights_begin(begin_msg)

            offset = 0
            tensor_bytes = t["tensor_bytes"]
            while offset < len(tensor_bytes):
                chunk = tensor_bytes[offset : offset + chunk_size]
                chunk_msg = {
                    "expert_id": int(expert_id),
                    "worker_id": int(target["worker_id"]),
                    "tensor_kind": t["tensor_kind"],
                    "chunk_offset": offset,
                    "chunk_data": chunk,
                }
                client.send_load_weights_chunk_oneway(chunk_msg)
                offset += len(chunk)

            end_msg = {
                "expert_id": int(expert_id),
                "worker_id": int(target["worker_id"]),
                "tensor_kind": t["tensor_kind"],
            }
            client.send_load_weights_end(end_msg)


def _resident_set(
    coord: Coordinator,
    node_instance_id: str,
    worker_id: int,
) -> Set[int]:
    return set(
        int(x)
        for x in coord.node_resident_inventories
        .get(str(node_instance_id), {})
        .get(int(worker_id), set())
    )


def _assert_ack_ok(acks: List[dict]) -> None:
    for ack in acks:
        if int(ack["status_code"]) != 0:
            raise RuntimeError(
                f"placement ack error: node={ack['node_instance_id']} "
                f"status_code={ack['status_code']}"
            )


def _node_drop_map(coord: Coordinator, value: bool) -> Dict[str, bool]:
    return {
        str(node["node_instance_id"]): bool(value)
        for node in coord.node_inventories
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"], log_level=cfg["log_level"])

    model = cfg["model"]
    chunk_size = int(model["chunk_size"])
    expert_mem_bytes = int(model["expert_mem_bytes"])
    memory_utilization = float(model["memory_utilization"])

    # Keep this test small and deterministic.
    num_test_experts = 2

    # Use fixed ids far away from "normal" ids to reduce collision chance with
    # previous experiments on the same node.
    expert_ids_round1 = [9100001, 9100002]
    expert_ids_round2 = [9100101, 9100102]

    print("[e2e-placement] discover initial state")
    coord.discover_nodes()
    coord.print_summary()

    if not coord.gpu_inventory:
        raise RuntimeError("no workers discovered")

    #
    # Build a stable placement shape using dummy ids that are guaranteed not to
    # hit current residents, so the worker targets are determined purely by
    # fresh-placement logic.
    #
    coord.discover_and_build_placement(
        expert_ids=expert_ids_round1,
        expert_mem_bytes=expert_mem_bytes,
        memory_utilization=memory_utilization,
        allow_drop_non_target_residents=False,
    )

    if len(coord.placements) != num_test_experts:
        raise RuntimeError(
            f"expected {num_test_experts} placements, got {len(coord.placements)}"
        )

    print("[e2e-placement] initial placement targets")
    coord.print_placement()

    target_keys: List[Tuple[str, int]] = [
        (str(p["node_instance_id"]), int(p["worker_id"])) for p in coord.placements
    ]

    #
    # Round 1: place A, keep non-target residents, then upload A.
    #
    coord.drop_non_target_residents_by_node = _node_drop_map(coord, False)
    print("[e2e-placement] round1 send placement drop_non_target_residents=0")
    acks = coord.send_placement_plan()
    _assert_ack_ok(acks)

    for target in coord.placements:
        _upload_dummy_expert(
            target=target,
            expert_id=int(target["expert_id"]),
            chunk_size=chunk_size,
            log_level=cfg["log_level"],
        )

    coord.discover_nodes()

    for target in coord.placements:
        node_instance_id = str(target["node_instance_id"])
        worker_id = int(target["worker_id"])
        expert_id = int(target["expert_id"])
        resident = _resident_set(coord, node_instance_id, worker_id)
        if expert_id not in resident:
            raise RuntimeError(
                f"round1 resident missing after upload: "
                f"node={node_instance_id} worker={worker_id} expert={expert_id}"
            )

    print("[e2e-placement] round1 uploaded experts are resident")

    #
    # Round 2: place B on the same targets, but do NOT drop non-target residents.
    # A should still remain resident because we only changed the target plan.
    #
    for p, eid in zip(coord.placements, expert_ids_round2):
        p["expert_id"] = int(eid)

    coord.drop_non_target_residents_by_node = _node_drop_map(coord, False)
    print("[e2e-placement] round2 send placement drop_non_target_residents=0")
    acks = coord.send_placement_plan()
    _assert_ack_ok(acks)

    coord.discover_nodes()

    for (node_instance_id, worker_id), old_expert_id, new_expert_id in zip(
        target_keys,
        expert_ids_round1,
        expert_ids_round2,
    ):
        resident = _resident_set(coord, node_instance_id, worker_id)

        if int(old_expert_id) not in resident:
            raise RuntimeError(
                f"round2 keep-path failed: "
                f"old expert disappeared with drop_non_target_residents=0 "
                f"node={node_instance_id} worker={worker_id} expert={old_expert_id}"
            )

        if int(new_expert_id) in resident:
            raise RuntimeError(
                f"round2 unexpected resident hit: "
                f"new target expert appeared before upload "
                f"node={node_instance_id} worker={worker_id} expert={new_expert_id}"
            )

    print("[e2e-placement] keep-path verified: old residents preserved and new targets not auto-loaded")

    #
    # Round 3: place B again, but now DROP non-target residents.
    # A should disappear immediately even though B has not been loaded yet.
    #
    coord.drop_non_target_residents_by_node = _node_drop_map(coord, True)
    print("[e2e-placement] round3 send placement drop_non_target_residents=1")
    acks = coord.send_placement_plan()
    _assert_ack_ok(acks)

    coord.discover_nodes()

    for (node_instance_id, worker_id), old_expert_id in zip(target_keys, expert_ids_round1):
        resident = _resident_set(coord, node_instance_id, worker_id)
        if int(old_expert_id) in resident:
            raise RuntimeError(
                f"round3 drop-path failed: "
                f"old expert still resident with drop_non_target_residents=1 "
                f"node={node_instance_id} worker={worker_id} expert={old_expert_id}"
            )

    print("[e2e-placement] drop-path verified: old residents removed")
    print("PASS=1")


if __name__ == "__main__":
    main()
