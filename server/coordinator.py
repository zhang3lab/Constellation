from typing import Any, Dict, List

from server.client import NodeClient
from server.placement import build_first_fit_placement


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
                local_gpu_id = gpu["local_gpu_id"]
                gpu_uid_global = f"{node_instance_id}/gpu{local_gpu_id}"

                row = {
                    "node_instance_id": node_instance_id,
                    "reported_node_id": inv["node_id"],
                    "host": host,
                    "control_port": control_port,

                    "gpu_uid_global": gpu_uid_global,
                    "gpu_uid_reported": gpu["gpu_uid"],

                    "local_gpu_id": local_gpu_id,
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
                f"local_gpu_id={gpu['local_gpu_id']} "
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
        self.placements = build_first_fit_placement(
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
                f"local_gpu_id={p['local_gpu_id']} "
                f"worker_port={p['worker_port']} "
                f"gpu_name={p['gpu_name']} "
                f"expert_mem={gib:.2f}GiB"
            )
