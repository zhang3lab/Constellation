from __future__ import annotations

import argparse
import base64
import time

import cupy as cp
from cupy.cuda import runtime as curuntime

from server.manual_compare.cache_client import CacheClient


class BorrowedCudaTensor:
    def __init__(self, ptr: int, nbytes: int, shape: tuple[int, ...], dtype: str, device_id: int):
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)
        self.shape = tuple(shape)
        self.dtype = cp.dtype(dtype)
        self.device_id = int(device_id)

        with cp.cuda.Device(self.device_id):
            self.mem = cp.cuda.UnownedMemory(self.ptr, self.nbytes, self, device_id=self.device_id)
            self.memptr = cp.cuda.MemoryPointer(self.mem, 0)
            self.array = cp.ndarray(self.shape, dtype=self.dtype, memptr=self.memptr)

    def close(self) -> None:
        curuntime.ipcCloseMemHandle(self.ptr)


def open_borrowed_tensor(desc: dict, device_id: int) -> BorrowedCudaTensor:
    handle = base64.b64decode(desc["handle_b64"])
    ptr = curuntime.ipcOpenMemHandle(handle)
    return BorrowedCudaTensor(
        ptr=ptr,
        nbytes=int(desc["nbytes"]),
        shape=tuple(int(x) for x in desc["shape"]),
        dtype=str(desc["dtype"]),
        device_id=int(device_id),
    )


def silu(x: cp.ndarray) -> cp.ndarray:
    return x / (1.0 + cp.exp(-x))


def sync_device(device_id: int) -> None:
    with cp.cuda.Device(device_id):
        cp.cuda.runtime.deviceSynchronize()


def run_one_expert(x_fp: cp.ndarray, w_up: cp.ndarray, w_gate: cp.ndarray, w_down: cp.ndarray) -> cp.ndarray:
    gate_linear = x_fp @ w_gate.T
    up_linear = x_fp @ w_up.T
    act = silu(gate_linear.astype(cp.float32)).astype(x_fp.dtype)
    mul = act * up_linear
    out = mul @ w_down.T
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47000)
    ap.add_argument("--layer-id", type=int, default=3)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--expert-ids", type=str, required=True, help="comma-separated, e.g. 18,29,34,43,235,254,255,226")
    ap.add_argument("--weights", type=str, required=True, help="comma-separated, same length as expert-ids")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    expert_ids = [int(x.strip()) for x in args.expert_ids.split(",") if x.strip()]
    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    if len(expert_ids) != len(weights):
        raise ValueError("expert-ids and weights length mismatch")

    hidden = 7168
    lease_ids: list[str] = []
    opened: list[tuple[BorrowedCudaTensor, BorrowedCudaTensor, BorrowedCudaTensor]] = []

    with CacheClient(args.host, args.port) as client:
        t0 = time.perf_counter()
        resp = client.borrow_expert_batch(
            layer_id=args.layer_id,
            expert_ids=expert_ids,
            device_id=args.device_id,
        )
        t1 = time.perf_counter()
        assert resp["ok"], resp

        print(f"[borrow_batch] n={len(expert_ids)} wall_ms={(t1 - t0) * 1000.0:.3f}")

        try:
            for item in resp["items"]:
                lease_ids.append(item["lease_id"])
                tensors = item["tensors"]

                w_up = open_borrowed_tensor(tensors["w_up"], args.device_id)
                w_gate = open_borrowed_tensor(tensors["w_gate"], args.device_id)
                w_down = open_borrowed_tensor(tensors["w_down"], args.device_id)
                opened.append((w_up, w_gate, w_down))

            with cp.cuda.Device(args.device_id):
                x = cp.random.randn(int(args.batch_size), hidden, dtype=cp.float32)
                x_fp = x.astype(cp.float16)

                sync_device(args.device_id)
                t2 = time.perf_counter()

                routed = cp.zeros((int(args.batch_size), hidden), dtype=cp.float16)
                for idx, (w_up, w_gate, w_down) in enumerate(opened):
                    out = run_one_expert(x_fp, w_up.array, w_gate.array, w_down.array)
                    routed = routed + out * cp.float16(weights[idx])

                sync_device(args.device_id)
                t3 = time.perf_counter()

                print(f"[topk_forward] wall_ms={(t3 - t2) * 1000.0:.3f}")
                print("routed shape:", routed.shape)
                print("routed dtype:", routed.dtype)
                print("routed sum_fp32:", float(routed.astype(cp.float32).sum().item()))
                print("routed mean_fp32:", float(routed.astype(cp.float32).mean().item()))

        finally:
            for w_up, w_gate, w_down in opened:
                w_up.close()
                w_gate.close()
                w_down.close()
            if lease_ids:
                print(client.return_expert_batch(lease_ids))


if __name__ == "__main__":
    main()
