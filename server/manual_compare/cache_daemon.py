from __future__ import annotations

import argparse
import base64
import json
import socketserver
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import cupy as cp
from cupy.cuda import runtime as curuntime

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
    pin_count: int = 0
    lease_ids: set[str] = field(default_factory=set)
    last_access_ts: float = 0.0
    cupy_arrays: dict[str, cp.ndarray] = field(default_factory=dict)


class CacheDaemon:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._experts: dict[tuple[int, int, int], ExpertResident] = {}
        self._leases: dict[str, tuple[int, int, int]] = {}

    def _export_ipc_handle(self, arr: cp.ndarray) -> bytes:
        ptr = int(arr.data.ptr)
        return curuntime.ipcGetMemHandle(ptr)

    def _make_tensor_resident(self, name: str, arr: cp.ndarray, device_id: int) -> TensorResident:
        return TensorResident(
            name=name,
            ptr=int(arr.data.ptr),
            shape=tuple(int(x) for x in arr.shape),
            dtype=str(arr.dtype),
            nbytes=int(arr.nbytes),
            device_id=int(device_id),
            ipc_handle=self._export_ipc_handle(arr),
        )

    def _ensure_expert_on_device(
        self,
        layer_id: int,
        expert_id: int,
        device_id: int,
    ) -> ExpertResident:
        key = (int(layer_id), int(expert_id), int(device_id))

        with self._lock:
            cached = self._experts.get(key)
            if cached is not None:
                cached.last_access_ts = time.time()
                return cached

        # 第一版：假 expert。后面你把这里替换成真实 load/dequant/copy。
        with cp.cuda.Device(device_id):
            w_up = cp.empty((2048, 7168), dtype=cp.float16)
            w_gate = cp.empty((2048, 7168), dtype=cp.float16)
            w_down = cp.empty((7168, 2048), dtype=cp.float16)

            # 给点可见内容，方便调试
            w_up.fill(1)
            w_gate.fill(2)
            w_down.fill(3)

            ex = ExpertResident(
                layer_id=layer_id,
                expert_id=expert_id,
                device_id=device_id,
                tensors={
                    "w_up": self._make_tensor_resident("w_up", w_up, device_id),
                    "w_gate": self._make_tensor_resident("w_gate", w_gate, device_id),
                    "w_down": self._make_tensor_resident("w_down", w_down, device_id),
                },
                pin_count=0,
                lease_ids=set(),
                last_access_ts=time.time(),
                cupy_arrays={
                    "w_up": w_up,
                    "w_gate": w_gate,
                    "w_down": w_down,
                },
            )

        with self._lock:
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
            self._leases[lease_id] = (int(layer_id), int(expert_id), int(device_id))

            return {
                "ok": True,
                "lease_id": lease_id,
                "layer_id": int(layer_id),
                "expert_id": int(expert_id),
                "device_id": int(device_id),
                "tensors": {
                    name: {
                        "handle_b64": base64.b64encode(t.ipc_handle).decode("ascii"),
                        "shape": list(t.shape),
                        "dtype": t.dtype,
                        "nbytes": t.nbytes,
                    }
                    for name, t in ex.tensors.items()
                },
                "pin_count": ex.pin_count,
            }

    def return_expert(self, lease_id: str) -> dict[str, Any]:
        with self._lock:
            key = self._leases.pop(lease_id, None)
            if key is None:
                return {"ok": False, "error": "unknown lease_id"}

            ex = self._experts[key]
            if lease_id in ex.lease_ids:
                ex.lease_ids.remove(lease_id)
            ex.pin_count = max(0, ex.pin_count - 1)
            ex.last_access_ts = time.time()

            return {
                "ok": True,
                "lease_id": lease_id,
                "pin_count": ex.pin_count,
            }

    def query(self) -> dict[str, Any]:
        with self._lock:
            items = []
            for (layer_id, expert_id, device_id), ex in self._experts.items():
                items.append(
                    {
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "device_id": device_id,
                        "pin_count": ex.pin_count,
                        "lease_count": len(ex.lease_ids),
                        "last_access_ts": ex.last_access_ts,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47000)
    args = ap.parse_args()

    daemon = CacheDaemon()
    CacheRequestHandler.daemon_ref = daemon

    class ThreadedTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    with ThreadedTCPServer((args.host, args.port), CacheRequestHandler) as server:
        print(f"[cache-daemon] listening on {args.host}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
