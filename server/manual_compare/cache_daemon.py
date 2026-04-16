from __future__ import annotations

import argparse
import base64
import json
import socketserver
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

import cupy as cp
import torch
from cupy.cuda import runtime as curuntime

from server.deepseek_model_loader import DeepseekModelLoader


cp.cuda.set_allocator(None)
cp.cuda.set_pinned_memory_allocator(None)


def recv_json_line(rfile) -> dict[str, Any]:
    line = rfile.readline()
    if not line:
        raise EOFError("connection closed")
    return json.loads(line.decode("utf-8"))


def send_json_line(wfile, obj: dict[str, Any]) -> None:
    payload = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    wfile.write(payload)
    wfile.flush()


def torch_tensor_to_cupy_on_device(
    x: torch.Tensor,
    device_id: int,
    target_torch_dtype: torch.dtype,
) -> cp.ndarray:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(x).__name__}")

    x_cuda = x.detach().contiguous().to(
        device=f"cuda:{int(device_id)}",
        dtype=target_torch_dtype,
    )
    return cp.from_dlpack(x_cuda)


@dataclass
class TensorResident:
    name: str
    ptr: int
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    device_id: int
    ipc_handle: bytes


@dataclass
class CpuResident:
    layer_id: int
    expert_id: int
    w_up_cpu: torch.Tensor
    w_gate_cpu: torch.Tensor
    w_down_cpu: torch.Tensor
    last_access_ts: float = 0.0


@dataclass
class GpuResident:
    layer_id: int
    expert_id: int
    device_id: int
    tensors: dict[str, TensorResident]
    cupy_arrays: dict[str, cp.ndarray]
    pin_count: int = 0
    lease_ids: set[str] = field(default_factory=set)
    last_access_ts: float = 0.0


class CacheDaemon:
    def __init__(
        self,
        model_dir: str,
        resident_dtype: str = "bfloat16",
    ) -> None:
        self._lock = threading.RLock()
        self._loader = DeepseekModelLoader(str(model_dir))

        self._cpu_experts: dict[tuple[int, int], CpuResident] = {}
        self._gpu_experts: dict[tuple[int, int, int], GpuResident] = {}
        self._leases: dict[str, tuple[int, int, int]] = {}

        # connection tracking
        self._conn_to_leases: dict[str, set[str]] = {}
        self._lease_to_conn: dict[str, str] = {}

        self._resident_dtype_str = str(resident_dtype)
        self._resident_torch_dtype = self._parse_torch_dtype(self._resident_dtype_str)
        self._resident_cupy_dtype = self._parse_cupy_dtype(self._resident_dtype_str)

    @staticmethod
    def _parse_torch_dtype(dtype_str: str) -> torch.dtype:
        key = dtype_str.lower()
        m = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if key not in m:
            raise ValueError(f"unsupported resident dtype: {dtype_str}")
        return m[key]

    @staticmethod
    def _parse_cupy_dtype(dtype_str: str):
        key = dtype_str.lower()
        if key in ("float16", "fp16"):
            return cp.dtype("float16")
        if key in ("bfloat16", "bf16"):
            return cp.dtype("bfloat16")
        if key in ("float32", "fp32"):
            return cp.dtype("float32")
        raise ValueError(f"unsupported resident dtype: {dtype_str}")

    @staticmethod
    def _cpu_key(layer_id: int, expert_id: int) -> tuple[int, int]:
        return (int(layer_id), int(expert_id))

    @staticmethod
    def _gpu_key(layer_id: int, expert_id: int, device_id: int) -> tuple[int, int, int]:
        return (int(layer_id), int(expert_id), int(device_id))

    def register_connection(self, connection_id: str) -> None:
        connection_id = str(connection_id)
        with self._lock:
            self._conn_to_leases.setdefault(connection_id, set())

    def _register_lease_to_connection(self, connection_id: str, lease_id: str) -> None:
        connection_id = str(connection_id)
        lease_id = str(lease_id)
        with self._lock:
            self._conn_to_leases.setdefault(connection_id, set()).add(lease_id)
            self._lease_to_conn[lease_id] = connection_id

    def _unregister_lease_from_connection(self, lease_id: str) -> None:
        lease_id = str(lease_id)
        with self._lock:
            conn_id = self._lease_to_conn.pop(lease_id, None)
            if conn_id is None:
                return
            s = self._conn_to_leases.get(conn_id)
            if s is not None:
                s.discard(lease_id)

    def on_client_disconnect(self, connection_id: str) -> None:
        connection_id = str(connection_id)

        with self._lock:
            lease_ids = list(self._conn_to_leases.get(connection_id, set()))

        if not lease_ids:
            with self._lock:
                self._conn_to_leases.pop(connection_id, None)
            return

        print(
            f"[cache-daemon] client disconnect cleanup "
            f"connection_id={connection_id} leases={len(lease_ids)}",
            flush=True,
        )

        self.return_expert_batch(lease_ids)

        with self._lock:
            self._conn_to_leases.pop(connection_id, None)

    def _export_ipc_handle(self, arr: cp.ndarray) -> bytes:
        ptr = int(arr.data.ptr)
        return curuntime.ipcGetMemHandle(ptr)

    def _make_tensor_resident(
        self,
        name: str,
        arr: cp.ndarray,
        device_id: int,
    ) -> TensorResident:
        return TensorResident(
            name=str(name),
            ptr=int(arr.data.ptr),
            shape=tuple(int(x) for x in arr.shape),
            dtype=str(arr.dtype),
            nbytes=int(arr.nbytes),
            device_id=int(device_id),
            ipc_handle=self._export_ipc_handle(arr),
        )

    def _ensure_expert_on_cpu(
        self,
        layer_id: int,
        expert_id: int,
    ) -> CpuResident:
        cpu_key = self._cpu_key(layer_id, expert_id)

        with self._lock:
            cached = self._cpu_experts.get(cpu_key)
            if cached is not None:
                cached.last_access_ts = time.time()
                return cached

        print(
            f"[cache-daemon] loading expert-to-cpu "
            f"layer={layer_id} expert={expert_id} dtype={self._resident_dtype_str}",
            flush=True,
        )

        t0 = time.perf_counter()

        w_up_t, w_gate_t, w_down_t = self._loader.load_routed_expert_triplet_fp32(
            layer_id=int(layer_id),
            expert_id=int(expert_id),
        )

        w_up_t = w_up_t.to(dtype=self._resident_torch_dtype).cpu().contiguous()
        w_gate_t = w_gate_t.to(dtype=self._resident_torch_dtype).cpu().contiguous()
        w_down_t = w_down_t.to(dtype=self._resident_torch_dtype).cpu().contiguous()

        cpu_ex = CpuResident(
            layer_id=int(layer_id),
            expert_id=int(expert_id),
            w_up_cpu=w_up_t,
            w_gate_cpu=w_gate_t,
            w_down_cpu=w_down_t,
            last_access_ts=time.time(),
        )

        t1 = time.perf_counter()

        print(
            f"[cache-daemon] loaded expert-to-cpu "
            f"layer={layer_id} expert={expert_id} "
            f"disk_and_cast_ms={(t1 - t0) * 1000.0:.3f}",
            flush=True,
        )

        with self._lock:
            cached = self._cpu_experts.get(cpu_key)
            if cached is not None:
                return cached
            self._cpu_experts[cpu_key] = cpu_ex
            return cpu_ex

    def _promote_expert_to_device(
        self,
        layer_id: int,
        expert_id: int,
        device_id: int,
    ) -> GpuResident:
        gpu_key = self._gpu_key(layer_id, expert_id, device_id)

        with self._lock:
            cached = self._gpu_experts.get(gpu_key)
            if cached is not None:
                cached.last_access_ts = time.time()
                return cached

        cpu_ex = self._ensure_expert_on_cpu(layer_id, expert_id)

        print(
            f"[cache-daemon] promote expert-to-gpu "
            f"layer={layer_id} expert={expert_id} device=cuda:{device_id}",
            flush=True,
        )

        t0 = time.perf_counter()

        w_up = torch_tensor_to_cupy_on_device(
            cpu_ex.w_up_cpu,
            device_id=device_id,
            target_torch_dtype=self._resident_torch_dtype,
        )
        w_gate = torch_tensor_to_cupy_on_device(
            cpu_ex.w_gate_cpu,
            device_id=device_id,
            target_torch_dtype=self._resident_torch_dtype,
        )
        w_down = torch_tensor_to_cupy_on_device(
            cpu_ex.w_down_cpu,
            device_id=device_id,
            target_torch_dtype=self._resident_torch_dtype,
        )

        gpu_ex = GpuResident(
            layer_id=int(layer_id),
            expert_id=int(expert_id),
            device_id=int(device_id),
            tensors={
                "w_up": self._make_tensor_resident("w_up", w_up, device_id),
                "w_gate": self._make_tensor_resident("w_gate", w_gate, device_id),
                "w_down": self._make_tensor_resident("w_down", w_down, device_id),
            },
            cupy_arrays={
                "w_up": w_up,
                "w_gate": w_gate,
                "w_down": w_down,
            },
            pin_count=0,
            lease_ids=set(),
            last_access_ts=time.time(),
        )

        t1 = time.perf_counter()

        print(
            f"[cache-daemon] promoted expert-to-gpu "
            f"layer={layer_id} expert={expert_id} device=cuda:{device_id} "
            f"h2d_ms={(t1 - t0) * 1000.0:.3f} "
            f"resident_dtypes=({w_up.dtype}, {w_gate.dtype}, {w_down.dtype})",
            flush=True,
        )

        with self._lock:
            cached = self._gpu_experts.get(gpu_key)
            if cached is not None:
                return cached
            self._gpu_experts[gpu_key] = gpu_ex
            return gpu_ex

    def _ensure_expert_on_device(
        self,
        layer_id: int,
        expert_id: int,
        device_id: int,
    ) -> GpuResident:
        return self._promote_expert_to_device(layer_id, expert_id, device_id)

    def borrow_expert_batch(
        self,
        layer_id: int,
        expert_ids: list[int],
        device_id: int,
        connection_id: str,
    ) -> dict[str, Any]:
        layer_id = int(layer_id)
        device_id = int(device_id)
        connection_id = str(connection_id)
        expert_ids = [int(x) for x in expert_ids]

        self.register_connection(connection_id)

        items = []
        residents: list[tuple[int, GpuResident]] = []

        for expert_id in expert_ids:
            with self._lock:
                if connection_id not in self._conn_to_leases:
                    raise RuntimeError(
                        f"connection {connection_id} is gone during borrow_expert_batch"
                    )

            ex = self._ensure_expert_on_device(layer_id, expert_id, device_id)
            residents.append((expert_id, ex))

        with self._lock:
            for expert_id, ex in residents:
                lease_id = str(uuid.uuid4())
                ex.pin_count += 1
                ex.lease_ids.add(lease_id)
                ex.last_access_ts = time.time()

                self._leases[lease_id] = self._gpu_key(layer_id, expert_id, device_id)
                self._register_lease_to_connection(connection_id, lease_id)

                items.append(
                    {
                        "lease_id": lease_id,
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "device_id": device_id,
                        "pin_count": ex.pin_count,
                        "tensors": {
                            name: {
                                "handle_b64": base64.b64encode(t.ipc_handle).decode("ascii"),
                                "shape": list(t.shape),
                                "dtype": t.dtype,
                                "nbytes": t.nbytes,
                            }
                            for name, t in ex.tensors.items()
                        },
                    }
                )

        return {
            "ok": True,
            "layer_id": layer_id,
            "device_id": device_id,
            "items": items,
        }

    def return_expert_batch(self, lease_ids: list[str]) -> dict[str, Any]:
        lease_ids = [str(x) for x in lease_ids]
        out = []

        with self._lock:
            for lease_id in lease_ids:
                key = self._leases.pop(lease_id, None)
                if key is None:
                    out.append(
                        {
                            "ok": False,
                            "lease_id": lease_id,
                            "error": f"unknown lease_id: {lease_id}",
                        }
                    )
                    continue

                self._unregister_lease_from_connection(lease_id)

                ex = self._gpu_experts[key]
                if lease_id in ex.lease_ids:
                    ex.lease_ids.remove(lease_id)
                ex.pin_count = max(0, ex.pin_count - 1)
                ex.last_access_ts = time.time()

                out.append(
                    {
                        "ok": True,
                        "lease_id": lease_id,
                        "pin_count": ex.pin_count,
                    }
                )

                # 只从 VRAM 卸载，CPU resident 保留
                if ex.pin_count == 0:
                    print(
                        f"[cache-daemon] gpu-evict-on-return "
                        f"layer={key[0]} expert={key[1]} device=cuda:{key[2]}",
                        flush=True,
                    )
                    self._gpu_experts.pop(key, None)

        return {
            "ok": True,
            "items": out,
        }

    def query(self) -> dict[str, Any]:
        with self._lock:
            cpu_items = []
            for (layer_id, expert_id), ex in self._cpu_experts.items():
                cpu_items.append(
                    {
                        "layer_id": int(layer_id),
                        "expert_id": int(expert_id),
                        "last_access_ts": float(ex.last_access_ts),
                    }
                )

            gpu_items = []
            for (layer_id, expert_id, device_id), ex in self._gpu_experts.items():
                gpu_items.append(
                    {
                        "layer_id": int(layer_id),
                        "expert_id": int(expert_id),
                        "device_id": int(device_id),
                        "pin_count": int(ex.pin_count),
                        "lease_count": int(len(ex.lease_ids)),
                        "last_access_ts": float(ex.last_access_ts),
                        "tensor_names": sorted(ex.tensors.keys()),
                    }
                )

            return {
                "ok": True,
                "cpu_experts": cpu_items,
                "gpu_experts": gpu_items,
            }


class CacheRequestHandler(socketserver.StreamRequestHandler):
    daemon_ref: CacheDaemon = None  # type: ignore[assignment]

    def handle(self) -> None:
        connection_id = f"{self.client_address[0]}:{self.client_address[1]}:{id(self)}"
        self.daemon_ref.register_connection(connection_id)

        try:
            while True:
                try:
                    req = recv_json_line(self.rfile)
                except EOFError:
                    return

                op = req.get("op")
                try:
                    if op == "borrow_expert_batch":
                        resp = self.daemon_ref.borrow_expert_batch(
                            layer_id=int(req["layer_id"]),
                            expert_ids=[int(x) for x in req["expert_ids"]],
                            device_id=int(req["device_id"]),
                            connection_id=connection_id,
                        )
                    elif op == "return_expert_batch":
                        resp = self.daemon_ref.return_expert_batch(
                            lease_ids=[str(x) for x in req["lease_ids"]],
                        )
                    elif op == "query":
                        resp = self.daemon_ref.query()
                    else:
                        resp = {"ok": False, "error": f"unknown op: {op}"}
                except Exception as e:
                    traceback.print_exc()
                    resp = {
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }

                send_json_line(self.wfile, resp)

        finally:
            self.daemon_ref.on_client_disconnect(connection_id)


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47000)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--resident-dtype", type=str, default="bfloat16")
    args = ap.parse_args()

    daemon = CacheDaemon(
        model_dir=args.model_dir,
        resident_dtype=args.resident_dtype,
    )
    CacheRequestHandler.daemon_ref = daemon

    with ThreadedTCPServer((args.host, args.port), CacheRequestHandler) as server:
        print(
            f"[cache-daemon] listening on {args.host}:{args.port} "
            f"model_dir={args.model_dir} resident_dtype={args.resident_dtype}",
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
