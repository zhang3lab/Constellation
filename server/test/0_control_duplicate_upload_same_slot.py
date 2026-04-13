import argparse
import struct
import time

from common.protocol import TensorKind
from server.client import NodeClient
from server.config import load_config
from server.coordinator import Coordinator


def _pack_f32_list(xs):
    return b"".join(struct.pack("<f", float(x)) for x in xs)


def _make_dummy_expert_tensors():
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


def _upload_one_tensor(client, expert_id, worker_id, tensor):
    begin_msg = {
        "expert_id": int(expert_id),
        "worker_id": int(worker_id),
        "tensor_kind": tensor["tensor_kind"],
        "total_bytes": len(tensor["tensor_bytes"]),
        "meta": {
            "shape": [int(x) for x in tensor["shape"]],
            "dtype": str(tensor["dtype"]),
            "row_block": int(tensor["row_block"]),
            "col_block": int(tensor["col_block"]),
        },
    }
    client.send_load_weights_begin(begin_msg)

    chunk_msg = {
        "expert_id": int(expert_id),
        "worker_id": int(worker_id),
        "tensor_kind": tensor["tensor_kind"],
        "chunk_offset": 0,
        "chunk_data": tensor["tensor_bytes"],
    }
    client.send_load_weights_chunk_oneway(chunk_msg)

    end_msg = {
        "expert_id": int(expert_id),
        "worker_id": int(worker_id),
        "tensor_kind": tensor["tensor_kind"],
    }
    client.send_load_weights_end(end_msg)


def _resident_set(coord, node_instance_id, worker_id):
    return set(
        int(x)
        for x in coord.node_resident_inventories
        .get(str(node_instance_id), {})
        .get(int(worker_id), set())
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"], log_level=cfg["log_level"])

    model = cfg["model"]
    expert_mem_bytes = int(model["expert_mem_bytes"])
    memory_utilization = float(model["memory_utilization"])

    expert_id = 9900001

    coord.discover_nodes()
    coord.build_placement(
        num_experts=1,
        expert_mem_bytes=expert_mem_bytes,
        memory_utilization=memory_utilization,
    )

    if len(coord.placements) != 1:
        raise RuntimeError(f"expected exactly 1 placement, got {len(coord.placements)}")

    target = dict(coord.placements[0])
    target["expert_id"] = expert_id

    print("[dup-upload] target")
    coord.placements = [target]
    coord.print_placement()

    acks = coord.send_placement_plan(drop_non_target_residents=False)
    if len(acks) != 1 or int(acks[0]["status_code"]) != 0:
        raise RuntimeError(f"bad placement ack: {acks}")

    tensors = _make_dummy_expert_tensors()

    host = str(target["host"])
    control_port = int(target["control_port"])
    worker_id = int(target["worker_id"])
    node_instance_id = str(target["node_instance_id"])

    # First full upload: this should enqueue the background build on the final tensor.
    with NodeClient(host, control_port, log_level=cfg["log_level"]) as client:
        for tensor in tensors:
            _upload_one_tensor(client, expert_id, worker_id, tensor)

    # Immediately do a second full upload for the same (expert, worker).
    # This is the edge case: pending_build_key may still be present.
    with NodeClient(host, control_port, log_level=cfg["log_level"]) as client:
        for tensor in tensors:
            _upload_one_tensor(client, expert_id, worker_id, tensor)

    # Poll resident inventory a bit; eventual readiness is enough.
    deadline = time.time() + 5.0
    while True:
        coord.discover_nodes()
        resident = _resident_set(coord, node_instance_id, worker_id)
        if expert_id in resident:
            break
        if time.time() >= deadline:
            raise RuntimeError(
                f"expert did not become resident in time: "
                f"node={node_instance_id} worker={worker_id} expert={expert_id}"
            )
        time.sleep(0.1)

    print(
        f"[dup-upload] resident ready "
        f"node={node_instance_id} worker={worker_id} expert={expert_id}"
    )
    print("PASS=1")


if __name__ == "__main__":
    main()
