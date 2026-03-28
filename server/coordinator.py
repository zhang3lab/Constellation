from pathlib import Path
from typing import Any, Dict, List, Sequence

from common.protocol import TensorKind
from server.client import NodeClient
from server.expert_placement import (
    find_expert_placement,
    group_placements_by_control_endpoint,
)
from server.model_locator import DeepseekModelLocator
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
                gpu_uid_global = f"{node_instance_id}/worker{worker_id}"
                gpu_uid_reported = f"{inv['node_id']}/worker{worker_id}"

                row = {
                    "node_instance_id": node_instance_id,
                    "reported_node_id": inv["node_id"],
                    "host": host,
                    "control_port": control_port,

                    "gpu_uid_global": gpu_uid_global,
                    "gpu_uid_reported": gpu_uid_reported,

                    "worker_id": worker_id,
                    "gpu_name": gpu["gpu_name"],
                    "total_mem_bytes": gpu["total_mem_bytes"],
                    "free_mem_bytes": gpu["free_mem_bytes"],
                    "worker_port": gpu["worker_port"],
                    "gpu_status": gpu["gpu_status"],
                    "gpu_vendor": gpu["gpu_vendor"],
                    "capability_flags": gpu["capability_flags"],
                    "arch_name": gpu["arch_name"],
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


    def send_one_tensor_bytes(
        self,
        expert_id: int,
        tensor_kind: TensorKind,
        tensor_bytes: bytes,
        chunk_size: int,
        shape: Sequence[int],
        dtype: str,
        row_block: int,
        col_block: int,
        *,
        client,
        target,
        verbose: bool = False,
    ) -> None:
        if client is None:
            raise ValueError("client must not be None")
        if target is None:
            raise ValueError("target must not be None")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

        expert_id = int(expert_id)

        shape_list = [int(x) for x in shape]
        for d in shape_list:
            if d < 0:
                raise ValueError(f"shape dim must be >= 0, got {d}")

        dtype_map = {
            "torch.float32": "float32",
            "torch.float16": "float16",
            "torch.bfloat16": "bfloat16",
            "torch.float8_e4m3fn": "float8_e4m3fn",
        }
        dtype = dtype_map.get(str(dtype), str(dtype))
        if not dtype:
            raise ValueError("dtype must be non-empty")

        row_block = int(row_block)
        col_block = int(col_block)
        if row_block <= 0 or col_block <= 0:
            raise ValueError(
                f"row_block/col_block must be > 0, got {row_block}/{col_block}"
            )

        begin_msg = {
            "expert_id": expert_id,
            "worker_id": int(target["worker_id"]),
            "tensor_kind": tensor_kind,
            "total_bytes": len(tensor_bytes),
            "meta": {
                "shape": shape_list,
                "dtype": dtype,
                "row_block": row_block,
                "col_block": col_block,
            },
        }

        client.send_load_weights_begin(begin_msg)

        offset = 0
        while offset < len(tensor_bytes):
            chunk = tensor_bytes[offset : offset + chunk_size]
            chunk_msg = {
                "expert_id": expert_id,
                "worker_id": int(target["worker_id"]),
                "tensor_kind": tensor_kind,
                "chunk_offset": offset,
                "chunk_data": chunk,
            }
            client.send_load_weights_chunk(chunk_msg)
            offset += len(chunk)

        end_msg = {
            "expert_id": expert_id,
            "worker_id": int(target["worker_id"]),
            "tensor_kind": tensor_kind,
        }
        client.send_load_weights_end(end_msg)

        if verbose:
            print(
                f"sent tensor bytes to {target['node_instance_id']} "
                f"expert={expert_id} worker_id={target['worker_id']} "
                f"tensor_kind={tensor_kind.name} total_bytes={len(tensor_bytes)} "
                f"shape={tuple(shape_list)} dtype={dtype}"
            )


    def build_preload_manifest(self, locator, experts_per_layer: int = 256):
        order = [
            ("w_up", TensorKind.WUp, 0),
            ("w_up_scale", TensorKind.WUpScale, 1),
            ("w_gate", TensorKind.WGate, 2),
            ("w_gate_scale", TensorKind.WGateScale, 3),
            ("w_down", TensorKind.WDown, 4),
            ("w_down_scale", TensorKind.WDownScale, 5),
        ]

        manifest = []

        for target in self.placements:
            expert_id = int(target["expert_id"])
            layer_id = expert_id // experts_per_layer
            local_expert_id = expert_id % experts_per_layer

            for tensor_kind_name, tensor_kind_enum, tensor_order in order:
                if tensor_kind_name in ("w_up", "w_gate", "w_down"):
                    tensor_name, shard_path = locator.resolve_deepseek_tensor(
                        layer_id=layer_id,
                        expert_id=local_expert_id,
                        tensor_kind=tensor_kind_name,
                    )
                else:
                    base_kind = {
                        "w_up_scale": "w_up",
                        "w_gate_scale": "w_gate",
                        "w_down_scale": "w_down",
                    }[tensor_kind_name]
                    tensor_name, shard_path = locator.resolve_deepseek_scale_tensor(
                        layer_id=layer_id,
                        expert_id=local_expert_id,
                        tensor_kind=base_kind,
                    )
    
                manifest.append(
                    {
                        "expert_id": expert_id,
                        "target": target,
                        "tensor_kind_name": tensor_kind_name,
                        "tensor_kind_enum": tensor_kind_enum,
                        "tensor_order": tensor_order,
                        "tensor_name": tensor_name,
                        "shard_path": shard_path,
                    }
                )

        return manifest


    def sort_preload_manifest(self, manifest):
        return sorted(
            manifest,
            key=lambda x: (
                str(x["target"]["host"]),
                int(x["target"]["control_port"]),
                str(x["shard_path"]),
                int(x["expert_id"]),
                int(x["tensor_order"]),
            ),
        )


    def send_one_tensor_from_open_shard(
        self,
        *,
        shard_file,
        item,
        chunk_size: int,
        client,
    ) -> None:
        tensor_name = item["tensor_name"]
        target = item["target"]
        tensor_kind = item["tensor_kind_enum"]
        expert_id = int(item["expert_id"])

        tensor_bytes, shape, dtype = DeepseekModelLocator.load_tensor_from_open_shard(
            shard_file,
            tensor_name,
        )

        row_block = 128
        col_block = 128

        self.send_one_tensor_bytes(
            expert_id=expert_id,
            tensor_kind=tensor_kind,
            tensor_bytes=tensor_bytes,
            chunk_size=chunk_size,
            shape=shape,
            dtype=dtype,
            row_block=row_block,
            col_block=col_block,
            client=client,
            target=target,
            verbose=False,
        )


    def preload_all_placed_experts(
        self,
        locator,
        chunk_size: int,
        experts_per_layer: int = 256,
    ):
        manifest = self.build_preload_manifest(
            locator=locator,
            experts_per_layer=experts_per_layer,
        )
        manifest = self.sort_preload_manifest(manifest)

        total_entries = len(manifest)
        total_experts = len({int(x["expert_id"]) for x in manifest})

        print(
            f"preloading all placed experts: experts={total_experts} "
            f"tensor_entries={total_entries}"
        )

        current_node_key = None
        current_shard_path = None
        client = None
        shard_file = None

        done_entries = 0
        all_expert_ids = sorted({int(x["expert_id"]) for x in manifest})

        try:
            for item in manifest:
                target = item["target"]
                node_key = (str(target["host"]), int(target["control_port"]))
                shard_path = str(item["shard_path"])

                if node_key != current_node_key:
                    if shard_file is not None:
                        shard_file.__exit__(None, None, None)
                        shard_file = None
                    if client is not None:
                        client.__exit__(None, None, None)
                        client = None

                    client = NodeClient(target["host"], target["control_port"])
                    client.__enter__()
                    current_node_key = node_key
                    current_shard_path = None

                    print(
                        f"[preload] open control "
                        f"node={target['node_instance_id']} "
                        f"host={target['host']} port={target['control_port']}"
                    )

                if shard_path != current_shard_path:
                    if shard_file is not None:
                        shard_file.__exit__(None, None, None)

                    shard_file = locator.open_shard(shard_path)
                    shard_file.__enter__()
                    current_shard_path = shard_path

                    print(
                        f"[preload] open shard "
                        f"node={target['node_instance_id']} "
                        f"path={shard_path}"
                    )

                self.send_one_tensor_from_open_shard(
                    shard_file=shard_file,
                    item=item,
                    chunk_size=chunk_size,
                    client=client,
                )

                done_entries += 1
                if done_entries == 1 or done_entries % 32 == 0 or done_entries == total_entries:
                    print(
                        f"[preload] {done_entries}/{total_entries} "
                        f"expert={item['expert_id']} tensor={item['tensor_kind_name']}"
                    )

        finally:
            if shard_file is not None:
                shard_file.__exit__(None, None, None)
            if client is not None:
                client.__exit__(None, None, None)

        return all_expert_ids
