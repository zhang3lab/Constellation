from pathlib import Path
from typing import Any, Dict, List

from common.protocol import TensorKind
from server.client import NodeClient
from server.placement import build_balanced_placement


class Coordinator:
    def __init__(self, nodes: List[Dict[str, Any]]):
        self.nodes = nodes
        self.node_inventories: List[Dict[str, Any]] = []
        self.gpu_inventory: List[Dict[str, Any]] = []
        self.placements: List[Dict[str, Any]] = []

    def discover_nodes(self) -> None:
        self.node_inventories = []
        self.gpu_inventory = []

        for node_cfg in self.nodes:
            host = node_cfg["host"]
            control_port = node_cfg["control_port"]

            node_instance_id = f"{host}:{control_port}"

            client = NodeClient(host, control_port)
            with client:
                inv = client.request_inventory()

            node_row = {
                "node_instance_id": node_instance_id,
                "reported_node_id": inv["node_id"],
                "host": host,
                "control_port": control_port,
                "node_status": inv["node_status"],
                "num_gpus": inv["num_gpus"],
                "gpus": inv["gpus"],
            }
            self.node_inventories.append(node_row)

            for gpu in inv["gpus"]:
                worker_id = gpu["worker_id"]
                gpu_uid_global = f"{node_instance_id}/gpu{worker_id}"

                row = {
                    "node_instance_id": node_instance_id,
                    "reported_node_id": inv["node_id"],
                    "host": host,
                    "control_port": control_port,

                    "gpu_uid_global": gpu_uid_global,
                    "gpu_uid_reported": gpu["gpu_uid"],

                    "worker_id": worker_id,
                    "gpu_name": gpu["gpu_name"],
                    "total_mem_bytes": gpu["total_mem_bytes"],
                    "free_mem_bytes": gpu["free_mem_bytes"],
                    "worker_port": gpu["worker_port"],
                    "gpu_status": gpu["gpu_status"],
                }
                self.gpu_inventory.append(row)

    def print_summary(self) -> None:
        print(f"discovered {len(self.node_inventories)} nodes")
        print(f"discovered {len(self.gpu_inventory)} gpus total")

        for node in self.node_inventories:
            print(
                f"node_instance_id={node['node_instance_id']} "
                f"reported_node_id={node['reported_node_id']} "
                f"status={node['node_status']} "
                f"num_gpus={node['num_gpus']}"
            )

        for gpu in self.gpu_inventory:
            free_gib = gpu["free_mem_bytes"] / (1024 ** 3)
            total_gib = gpu["total_mem_bytes"] / (1024 ** 3)
            print(
                f"node_instance_id={gpu['node_instance_id']} "
                f"reported_node_id={gpu['reported_node_id']} "
                f"gpu_uid_global={gpu['gpu_uid_global']} "
                f"gpu_uid_reported={gpu['gpu_uid_reported']} "
                f"worker_id={gpu['worker_id']} "
                f"name={gpu['gpu_name']} "
                f"free={free_gib:.2f}GiB/{total_gib:.2f}GiB "
                f"worker_port={gpu['worker_port']} "
                f"status={gpu['gpu_status']}"
            )

    def build_placement(
        self,
        num_experts: int,
        expert_mem_bytes: int,
        memory_utilization: float = 0.9,
    ) -> None:
        self.placements = build_balanced_placement(
            gpu_inventory=self.gpu_inventory,
            num_experts=num_experts,
            expert_mem_bytes=expert_mem_bytes,
            memory_utilization=memory_utilization,
        )

    def print_placement(self) -> None:
        print(f"built placement for {len(self.placements)} experts")
        for p in self.placements:
            gib = p["expert_mem_bytes"] / (1024 ** 3)
            print(
                f"expert={p['expert_id']} "
                f"node_instance_id={p['node_instance_id']} "
                f"gpu_uid_global={p['gpu_uid_global']} "
                f"worker_id={p['worker_id']} "
                f"worker_port={p['worker_port']} "
                f"gpu_name={p['gpu_name']} "
                f"expert_mem={gib:.2f}GiB"
            )

    def group_placements_by_node(self):
        grouped = {}
     
        for p in self.placements:
            node_instance_id = p["node_instance_id"]
            grouped.setdefault(node_instance_id, []).append(
                {
                    "expert_id": p["expert_id"],
                    "worker_id": p["worker_id"],
                }
            )
     
        for node_instance_id in grouped:
            grouped[node_instance_id].sort(
                key=lambda x: (x["worker_id"], x["expert_id"])
            )
     
        return grouped

    def send_placement_plan(self) -> None:
        grouped = self.group_placements_by_node()
     
        for node in self.node_inventories:
            node_instance_id = node["node_instance_id"]
            host = node["host"]
            control_port = node["control_port"]
     
            assignments = grouped.get(node_instance_id, [])
     
            client = NodeClient(host, control_port)
            with client:
                client.send_placement_plan(assignments)
     
            print(
                f"sent placement to {node_instance_id} "
                f"reported_node_id={node['reported_node_id']} "
                f"assignments={len(assignments)}"
            )

    def test_send_load_weights_begin(self) -> None:
        if not self.placements:
            raise RuntimeError("placements are empty")
     
        p = self.placements[0]
     
        msg = {
            "expert_id": p["expert_id"],
            "worker_id": p["worker_id"],
            "tensor_kind": TensorKind.WUp,
            "total_bytes": 123456,
        }
     
        client = NodeClient(p["host"], p["control_port"])
        with client:
            client.send_load_weights_begin(msg)
     
        print(
            f"sent LoadWeightsBegin to {p['node_instance_id']} "
            f"expert={p['expert_id']} worker_id={p['worker_id']} "
            f"tensor_kind={msg['tensor_kind'].name} total_bytes={msg['total_bytes']}"
        )

    def test_send_full_load_sequence(self) -> None:
        if not self.placements:
            raise RuntimeError("placements are empty")
     
        p = self.placements[0]
     
        fake_chunk = b"hello_fake_weight_bytes"
        total_bytes = len(fake_chunk)
     
        begin_msg = {
            "expert_id": p["expert_id"],
            "worker_id": p["worker_id"],
            "tensor_kind": TensorKind.WUp,
            "total_bytes": total_bytes,
        }
     
        chunk_msg = {
            "expert_id": p["expert_id"],
            "worker_id": p["worker_id"],
            "tensor_kind": TensorKind.WUp,
            "chunk_offset": 0,
            "chunk_data": fake_chunk,
        }
     
        end_msg = {
            "expert_id": p["expert_id"],
            "worker_id": p["worker_id"],
            "tensor_kind": TensorKind.WUp,
        }
     
        client = NodeClient(p["host"], p["control_port"])
        with client:
            client.send_load_weights_begin(begin_msg)
            client.send_load_weights_chunk(chunk_msg)
            client.send_load_weights_end(end_msg)
     
        print(
            f"sent full load sequence to {p['node_instance_id']} "
            f"expert={p['expert_id']} worker_id={p['worker_id']} "
            f"tensor_kind={begin_msg['tensor_kind'].name} total_bytes={total_bytes}"
        )

    def send_one_tensor_bytes(
        self,
        expert_id: int,
        tensor_kind: TensorKind,
        tensor_bytes: bytes,
        chunk_size: int,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
     
        target = None
        for p in self.placements:
            if p["expert_id"] == expert_id:
                target = p
                break
     
        if target is None:
            raise RuntimeError(f"expert {expert_id} not found in placements")
     
        begin_msg = {
            "expert_id": expert_id,
            "worker_id": target["worker_id"],
            "tensor_kind": tensor_kind,
            "total_bytes": len(tensor_bytes),
        }
     
        client = NodeClient(target["host"], target["control_port"])
        with client:
            client.send_load_weights_begin(begin_msg)
     
            offset = 0
            while offset < len(tensor_bytes):
                chunk = tensor_bytes[offset : offset + chunk_size]
                chunk_msg = {
                    "expert_id": expert_id,
                    "worker_id": target["worker_id"],
                    "tensor_kind": tensor_kind,
                    "chunk_offset": offset,
                    "chunk_data": chunk,
                }
                client.send_load_weights_chunk(chunk_msg)
                offset += len(chunk)
     
            end_msg = {
                "expert_id": expert_id,
                "worker_id": target["worker_id"],
                "tensor_kind": tensor_kind,
            }
            client.send_load_weights_end(end_msg)
     
        print(
            f"sent tensor bytes to {target['node_instance_id']} "
            f"expert={expert_id} worker_id={target['worker_id']} "
            f"tensor_kind={tensor_kind.name} total_bytes={len(tensor_bytes)}"
        )

    def send_one_expert_sixpack(
        self,
        expert_id: int,
        tensor_loader,
        chunk_size: int,
    ) -> None:
        order = [
            ("w_up", TensorKind.WUp),
            ("w_up_scale", TensorKind.WUpScale),
            ("w_gate", TensorKind.WGate),
            ("w_gate_scale", TensorKind.WGateScale),
            ("w_down", TensorKind.WDown),
            ("w_down_scale", TensorKind.WDownScale),
        ]

        for tensor_kind_name, tensor_kind_enum in order:
            tensor_name, shard_path, tensor_bytes, shape, dtype = tensor_loader(
                expert_id,
                tensor_kind_name,
            )

            print(f"resolved tensor_name={tensor_name}")
            print(f"resolved shard_path={shard_path}")
            print(f"resolved shape={shape} dtype={dtype} total_bytes={len(tensor_bytes)}")

            self.send_one_tensor_bytes(
                expert_id=expert_id,
                tensor_kind=tensor_kind_enum,
                tensor_bytes=tensor_bytes,
                chunk_size=chunk_size,
            )

    def preload_all_placed_experts(self, tensor_loader, chunk_size: int):
        expert_ids = sorted({int(p["expert_id"]) for p in self.placements})
        print(f"preloading all placed experts: {expert_ids}")

        for expert_id in expert_ids:
            print(f"preloading expert={expert_id}")
            self.send_one_expert_sixpack(
                expert_id=expert_id,
                tensor_loader=tensor_loader,
                chunk_size=chunk_size,
            )

        return expert_ids
