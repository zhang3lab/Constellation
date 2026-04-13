from pathlib import Path
import threading
import traceback
from typing import Any, Dict, List, Sequence

import time
     

from common.protocol import TensorKind
from server.client import NodeClient
from server.deepseek_model_loader import DeepseekModelLoader
from server.expert_placement import (
    find_expert_placement,
    group_placements_by_control_endpoint,
)
from server.logging_utils import log1, log2
from server.placement import build_balanced_placement


class Coordinator:
    def __init__(self, nodes: List[Dict[str, Any]], log_level: int = 0):
        self.nodes = nodes
        self.log_level = int(log_level)
        self.node_inventories: List[Dict[str, Any]] = []
        self.gpu_inventory: List[Dict[str, Any]] = []
        self.placements: List[Dict[str, Any]] = []
        self.node_resident_inventories: Dict[str, Dict[int, set[int]]] = {}


    def discover_nodes(self) -> None:
        self.node_inventories = []
        self.gpu_inventory = []
        self.node_resident_inventories = {}
     
        for node_cfg in self.nodes:
            host = node_cfg["host"]
            control_port = node_cfg["control_port"]
     
            node_instance_id = f"{host}:{control_port}"
     
            client = NodeClient(host, control_port, log_level=self.log_level)
            with client:
                inv = client.request_inventory()
                resident = client.request_resident_inventory()
     
            if int(resident["num_workers"]) != len(resident["workers"]):
                raise RuntimeError(
                    f"resident inventory num_workers mismatch "
                    f"for node_instance_id={node_instance_id}: "
                    f"header={resident['num_workers']} "
                    f"decoded={len(resident['workers'])}"
                )
     
            inventory_worker_ids = {int(g["worker_id"]) for g in inv["gpus"]}
     
            resident_by_worker: Dict[int, set[int]] = {}
            for worker in resident["workers"]:
                wid = int(worker["worker_id"])
     
                if wid not in inventory_worker_ids:
                    raise RuntimeError(
                        f"resident inventory returned unknown worker_id={wid} "
                        f"for node_instance_id={node_instance_id}"
                    )
     
                if wid in resident_by_worker:
                    raise RuntimeError(
                        f"resident inventory returned duplicate worker_id={wid} "
                        f"for node_instance_id={node_instance_id}"
                    )
     
                resident_by_worker[wid] = {
                    int(expert_id) for expert_id in worker["expert_ids"]
                }
     
            self.node_resident_inventories[node_instance_id] = resident_by_worker
     
            node_row = {
                "node_instance_id": node_instance_id,
                "reported_node_id": inv["node_id"],
                "host": host,
                "control_port": control_port,
                "node_status": inv["node_status"],
                "num_gpus": inv["num_gpus"],
                "gpus": inv["gpus"],
                "resident_inventory": resident,
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
                    "resident_expert_ids": sorted(
                        resident_by_worker.get(int(worker_id), set())
                    ),
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

        for node_instance_id, resident_by_worker in self.node_resident_inventories.items():
            total_resident = sum(len(x) for x in resident_by_worker.values())
            print(
                f"resident node_instance_id={node_instance_id} "
                f"workers={len(resident_by_worker)} "
                f"total_resident_experts={total_resident}"
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


    def send_placement_plan(self, drop_non_target_residents: bool = False) -> list[dict]:
        grouped = self.group_placements_by_node()
        acks = []
     
        for node in self.node_inventories:
            node_instance_id = node["node_instance_id"]
            host = node["host"]
            control_port = node["control_port"]
     
            assignments = grouped.get(node_instance_id, [])
     
            client = NodeClient(host, control_port, log_level=self.log_level)
            with client:
                ack = client.send_placement_plan(
                    assignments,
                    drop_non_target_residents=drop_non_target_residents,
                )
     
            ack = dict(ack)
            ack["node_instance_id"] = node_instance_id
            ack["reported_node_id"] = node["reported_node_id"]
            ack["host"] = host
            ack["control_port"] = control_port
            ack["num_assignments_sent"] = len(assignments)
            ack["drop_non_target_residents"] = bool(drop_non_target_residents)
            acks.append(ack)
     
            log1(
                self.log_level,
                f"sent placement to {node_instance_id} "
                f"reported_node_id={node['reported_node_id']} "
                f"assignments={len(assignments)} "
                f"drop_non_target_residents={int(drop_non_target_residents)} "
                f"needs_load={ack['needs_load']} "
                f"all_ready={ack['all_ready']} "
                f"target={ack['num_target_experts']} "
                f"ready={ack['num_ready_experts']}"
            )
     
        return acks


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
     
        t0 = time.perf_counter()
        client.send_load_weights_begin(begin_msg)
        t1 = time.perf_counter()
     
        chunk_count = 0
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
            client.send_load_weights_chunk_oneway(chunk_msg)
            offset += len(chunk)
            chunk_count += 1
     
        t2 = time.perf_counter()
     
        end_msg = {
            "expert_id": expert_id,
            "worker_id": int(target["worker_id"]),
            "tensor_kind": tensor_kind,
        }
        client.send_load_weights_end(end_msg)
        t3 = time.perf_counter()
     
        begin_ms = (t1 - t0) * 1000.0
        chunk_total_ms = (t2 - t1) * 1000.0
        end_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0
        chunk_avg_ms = (chunk_total_ms / chunk_count) if chunk_count > 0 else 0.0
     
        log2(
            self.log_level,
            f"[server-send-profile] "
            f"node={target['node_instance_id']} "
            f"expert={expert_id} "
            f"worker={target['worker_id']} "
            f"kind={tensor_kind.name} "
            f"bytes={len(tensor_bytes)} "
            f"shape={tuple(shape_list)} "
            f"dtype={dtype} "
            f"chunk_size={chunk_size} "
            f"chunks={chunk_count} "
            f"begin_ms={begin_ms:.3f} "
            f"chunk_total_ms={chunk_total_ms:.3f} "
            f"chunk_avg_ms={chunk_avg_ms:.3f} "
            f"end_ms={end_ms:.3f} "
            f"total_ms={total_ms:.3f}"
        )


    def build_preload_manifest(
        self,
        model_loader,
        experts_per_layer: int = 256,
        placements=None,
    ):
        order = [
            ("w_up", TensorKind.WUp, 0),
            ("w_up_scale", TensorKind.WUpScale, 1),
            ("w_gate", TensorKind.WGate, 2),
            ("w_gate_scale", TensorKind.WGateScale, 3),
            ("w_down", TensorKind.WDown, 4),
            ("w_down_scale", TensorKind.WDownScale, 5),
        ]
     
        if placements is None:
            placements = self.placements
     
        manifest = []
     
        for target in placements:
            expert_id = int(target["expert_id"])
            layer_id = expert_id // experts_per_layer
            local_expert_id = expert_id % experts_per_layer
     
            for tensor_kind_name, tensor_kind_enum, tensor_order in order:
                if tensor_kind_name in ("w_up", "w_gate", "w_down"):
                    tensor_name, shard_path = model_loader.resolve_deepseek_tensor(
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
                    tensor_name, shard_path = model_loader.resolve_deepseek_scale_tensor(
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
     
        t0 = time.perf_counter()
        tensor_bytes, shape, dtype = DeepseekModelLoader.load_tensor_from_open_shard(
            shard_file,
            item["tensor_name"],
        )
        t1 = time.perf_counter()
     
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
        )
        t2 = time.perf_counter()
     
        load_ms = (t1 - t0) * 1000.0
        send_ms = (t2 - t1) * 1000.0
        total_ms = (t2 - t0) * 1000.0
     
        log2(
            self.log_level,
            f"[server-upload-profile] "
            f"node={target['node_instance_id']} "
            f"expert={expert_id} "
            f"worker={target['worker_id']} "
            f"tensor={tensor_name} "
            f"kind={tensor_kind.name} "
            f"bytes={len(tensor_bytes)} "
            f"load_ms={load_ms:.3f} "
            f"send_ms={send_ms:.3f} "
            f"total_ms={total_ms:.3f}"
        )


    def _preload_manifest_for_one_node(
        self,
        model_loader,
        manifest_items,
        chunk_size: int,
        progress_prefix: str,
    ):
        if not manifest_items:
            return
     
        first_target = manifest_items[0]["target"]
        host = str(first_target["host"])
        control_port = int(first_target["control_port"])
        node_instance_id = str(first_target["node_instance_id"])
     
        client = None
        shard_file = None
        current_shard_path = None
        done_entries = 0
        total_entries = len(manifest_items)
     
        try:
            client = NodeClient(host, control_port, log_level=self.log_level)
            client.__enter__()
     
            log2(
                self.log_level,
                f"{progress_prefix} open control "
                f"node={node_instance_id} host={host} port={control_port}"
            )
     
            for item in manifest_items:
                shard_path = str(item["shard_path"])
     
                if shard_path != current_shard_path:
                    if shard_file is not None:
                        shard_file.__exit__(None, None, None)
                        shard_file = None
     
                    shard_file = model_loader.open_shard(shard_path)
                    shard_file.__enter__()
                    current_shard_path = shard_path
     
                    log2(
                        self.log_level,
                        f"{progress_prefix} open shard "
                        f"node={node_instance_id} path={shard_path}"
                    )
     
                self.send_one_tensor_from_open_shard(
                    shard_file=shard_file,
                    item=item,
                    chunk_size=chunk_size,
                    client=client,
                )
     
                done_entries += 1
                if done_entries == 1 or done_entries % 32 == 0 or done_entries == total_entries:
                    log2(
                        self.log_level,
                        f"{progress_prefix} {done_entries}/{total_entries} "
                        f"expert={item['expert_id']} tensor={item['tensor_kind_name']}"
                    )
     
        finally:
            if shard_file is not None:
                shard_file.__exit__(None, None, None)
            if client is not None:
                client.__exit__(None, None, None)


    def _placement_is_already_resident(self, placement: Dict[str, Any]) -> bool:
        node_instance_id = str(placement["node_instance_id"])
        worker_id = int(placement["worker_id"])
        expert_id = int(placement["expert_id"])

        resident_by_worker = self.node_resident_inventories.get(node_instance_id, {})
        return expert_id in resident_by_worker.get(worker_id, set())

    def preload_all_placed_experts(
        self,
        model_loader,
        chunk_size: int,
        experts_per_layer: int = 256,
        placement_acks=None,
    ):
        placements_for_preload = self.placements
     
        if placement_acks is not None:
            load_nodes = {
                str(ack["node_instance_id"])
                for ack in placement_acks
                if bool(ack["needs_load"])
            }
     
            for ack in placement_acks:
                if not bool(ack["needs_load"]) and bool(ack["all_ready"]):
                    log1(
                        self.log_level,
                        f"[preload] skip node={ack['node_instance_id']} "
                        f"target={ack['num_target_experts']} "
                        f"ready={ack['num_ready_experts']}"
                    )
     
            placements_for_preload = [
                p for p in self.placements
                if str(p["node_instance_id"]) in load_nodes
            ]
     
        all_expert_ids = sorted({int(p["expert_id"]) for p in self.placements})
     
        if not placements_for_preload:
            log1(self.log_level, "[preload] nothing to load")
            return all_expert_ids
     
        already_resident = []
        missing_resident = []
     
        for p in placements_for_preload:
            if self._placement_is_already_resident(p):
                already_resident.append(p)
            else:
                missing_resident.append(p)
     
        if already_resident:
            log1(
                self.log_level,
                f"[preload] already resident placements={len(already_resident)} "
                f"missing placements={len(missing_resident)}"
            )
     
        placements_for_preload = missing_resident
     
        if not placements_for_preload:
            log1(self.log_level, "[preload] everything already resident")
            return all_expert_ids
     
        manifest = self.build_preload_manifest(
            model_loader=model_loader,
            experts_per_layer=experts_per_layer,
            placements=placements_for_preload,
        )
        manifest = self.sort_preload_manifest(manifest)
     
        total_entries = len(manifest)
        experts_to_preload = len({int(x["expert_id"]) for x in manifest})

        log1(
            self.log_level,
            f"preloading all placed experts: experts={experts_to_preload} "
            f"tensor_entries={total_entries}"
        )
        log1(
            self.log_level,
            f"[preload] target_experts={len(all_expert_ids)} "
            f"experts_to_preload={experts_to_preload}"
        )
     
        jobs_by_node = {}
        for item in manifest:
            node_instance_id = str(item["target"]["node_instance_id"])
            jobs_by_node.setdefault(node_instance_id, []).append(item)
     
        log1(self.log_level, f"[preload] parallel nodes={len(jobs_by_node)}")
     
        errors = []
        err_lock = threading.Lock()
        threads = []
     
        def worker(node_instance_id, items):
            target = items[0]["target"]
            progress_prefix = f"[preload:{node_instance_id}]"
            try:
                self._preload_manifest_for_one_node(
                    model_loader=model_loader,
                    manifest_items=items,
                    chunk_size=chunk_size,
                    progress_prefix=progress_prefix,
                )
            except Exception:
                tb = traceback.format_exc()
                with err_lock:
                    errors.append(
                        f"node={node_instance_id} "
                        f"host={target['host']} port={target['control_port']}\n{tb}"
                    )
     
        for node_instance_id, items in jobs_by_node.items():
            th = threading.Thread(
                target=worker,
                args=(node_instance_id, items),
                name=f"preload-{node_instance_id}",
            )
            th.start()
            threads.append(th)
     
        for th in threads:
            th.join()
     
        if errors:
            for err in errors:
                print(f"[preload] ERROR\n{err}")
            raise RuntimeError(f"preload failed on {len(errors)} node(s)")

        self.discover_nodes()

        missing_after_preload = [
            p for p in placements_for_preload
            if not self._placement_is_already_resident(p)
        ]
        loaded_count = len(placements_for_preload) - len(missing_after_preload)

        log1(
            self.log_level,
            f"[preload] resident check loaded={loaded_count} "
            f"expected={len(placements_for_preload)} "
            f"missing={len(missing_after_preload)}"
        )

        if missing_after_preload:
            for p in missing_after_preload[:8]:
                print(
                    f"[preload] MISSING "
                    f"node={p['node_instance_id']} "
                    f"worker={p['worker_id']} "
                    f"expert={p['expert_id']}"
                )
            raise RuntimeError(
                f"preload resident check failed: "
                f"missing={len(missing_after_preload)}/"
                f"{len(placements_for_preload)}"
            )

        return all_expert_ids
