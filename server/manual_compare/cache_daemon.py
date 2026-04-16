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


# 避免 CuPy memory pool/suballocation 干扰 IPC handle。
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


def torch_cpu_tensor_to_cupy(x: torch.Tensor, device_id: int) -> cp.ndarray:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(x).__name__}")
    x = x.detach().cpu().contiguous()
    np_arr = x.numpy()
    with cp.cuda.Device(device_id):
        return cp.asarray(np_arr)


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
class ExpertResident:
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
        resident_dtype: str = "float16",
    ) -> None:
        self._lock = threading.RLock()
        self._loader = DeepseekModelLoader(str(model_dir))
        self._experts: dict[tuple[int, int, int], ExpertResident] = {}
        self._leases: dict[str, tuple[int, int, int]] = {}

        self._resident_dtype_str = str(resident_dtype)
        self._resident_torch_dtype = self._parse_torch_dtype(self._resident_dtype_str)
        self._resident_cupy_dtype = self._parse_cupy_dtype(self._resident_dtype_str)

    @staticmethod
    def _parse_torch_dtype(dtype_str: str) -> torch.dtype:
        m = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        key = dtype_str.lower()
        if key not in m:
            raise ValueError(f"unsupported resident dtype: {dtype_str}")
        return m[key]

    @staticmethod
    def _parse_cupy_dtype(dtype_str: str):
        m = {
            "float16": cp.float16,
            "fp16": cp.float16,
            "bfloat16": cp.float16,  # CuPy bf16 support depends on build; first版用 fp16 更稳
            "bf16": cp.float16,
            "float32": cp.float32,
            "fp32": cp.float32,
        }
        key = dtype_str.lower()
        if key not in m:
            raise ValueError(f"unsupported resident dtype: {dtype_str}")
        return m[key]

    @staticmethod
    def _key(layer_id: int, expert_id: int, device_id: int) -> tuple[int, int, int]:
        return (int(layer_id), int(expert_id), int(device_id))

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

    def _load_real_expert_to_device(
        self,
        layer_id: int,
        expert_id: int,
        device_id: int,
    ) -> ExpertResident:
        print(
            f"[cache-daemon] loading expert layer={layer_id} expert={expert_id} "
            f"device=cuda:{device_id} dtype={self._resident_dtype_str}",
            flush=True,
        )

        t0 = time.perf_counter()
        w_up_t, w_gate_t, w_down_t = self._loader.load_routed_expert_triplet_fp32(
            layer_id=int(layer_id),
            expert_id=int(expert_id),
        )
        t1 = time.perf_counter()

        w_up_t = w_up_t.to(dtype=self._resident_torch_dtype).contiguous()
        w_gate_t = w_gate_t.to(dtype=self._resident_torch_dtype).contiguous()
        w_down_t = w_down_t.to(dtype=self._resident_torch_dtype).contiguous()

        w_up = torch_cpu_tensor_to_cupy(w_up_t, device_id)
        w_gate = torch_cpu_tensor_to_cupy(w_gate_t, device_id)
        w_down = torch_cpu_tensor_to_cupy(w_down_t, device_id)
        t2 = time.perf_counter()

        ex = ExpertResident(
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
        t3 = time.perf_counter()

        print(
            f"[cache-daemon] loaded expert layer={layer_id} expert={expert_id} "
            f"device=cuda:{device_id} "
            f"disk_ms={(t1 - t0) * 1000.0:.3f} "
            f"h2d_ms={(t2 - t1) * 1000.0:.3f} "
            f"ipc_ms={(t3 - t2) * 1000.0:.3f}",
            flush=True,
        )
        return ex

    def _ensure_expert_on_device(
        self,
        layer_id: int,
        expert_id: int,
        device_id: int,
    ) -> ExpertResident:
        key = self._key(layer_id, expert_id, device_id)

        with self._lock:
            cached = self._experts.get(key)
            if cached is not None:
                cached.last_access_ts = time.time()
                return cached

        ex = self._load_real_expert_to_device(layer_id, expert_id, device_id)

        with self._lock:
            # double-check，防止并发重复 load
            cached = self._experts.get(key)
            if cached is not None:
                return cached
            self._experts[key] = ex
            return ex

    def borrow_expert(
        self,
        layer_id: int,
        expert_id: int,
        device_id: int,
    ) -> dict[str, Any]:
        ex = self._ensure_expert_on_device(layer_id, expert_id, device_id)

        lease_id = str(uuid.uuid4())
        with self._lock:
            ex.pin_count += 1
            ex.lease_ids.add(lease_id)
            ex.last_access_ts = time.time()
            self._leases[lease_id] = self._key(layer_id, expert_id, device_id)

            return {
                "ok": True,
                "lease_id": lease_id,
                "layer_id": int(layer_id),
                "expert_id": int(expert_id),
                "device_id": int(device_id),
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

    def return_expert(self, lease_id: str) -> dict[str, Any]:
        with self._lock:
            key = self._leases.pop(str(lease_id), None)
            if key is None:
                return {"ok": False, "error": f"unknown lease_id: {lease_id}"}

            ex = self._experts[key]
            if lease_id in ex.lease_ids:
                ex.lease_ids.remove(lease_id)
            ex.pin_count = max(0, ex.pin_count - 1)
            ex.last_access_ts = time.time()

            return {
                "ok": True,
                "lease_id": str(lease_id),
                "pin_count": ex.pin_count,
            }

    def query(self) -> dict[str, Any]:
        with self._lock:
            items = []
            for (layer_id, expert_id, device_id), ex in self._experts.items():
                items.append(
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
            return {"ok": True, "experts": items}


class CacheRequestHandler(socketserver.StreamRequestHandler):
    daemon_ref: CacheDaemon = None  # type: ignore[assignment]

    def handle(self) -> None:
        while True:
            try:
                req = recv_json_line(self.rfile)
            except EOFError:
                return

            op = req.get("op")
            try:
                if op == "borrow_expert":
                    resp = self.daemon_ref.borrow_expert(
                        layer_id=int(req["layer_id"]),
                        expert_id=int(req["expert_id"]),
                        device_id=int(req["device_id"]),
                    )
                elif op == "return_expert":
                    resp = self.daemon_ref.return_expert(
                        lease_id=str(req["lease_id"]),
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


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47000)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--resident-dtype", type=str, default="float16")
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
