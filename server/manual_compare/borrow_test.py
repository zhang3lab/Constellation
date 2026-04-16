from __future__ import annotations

import argparse
import base64

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
            self.mem = cp.cuda.UnownedMemory(
                self.ptr,
                self.nbytes,
                self,
                device_id=self.device_id,
            )
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47000)
    ap.add_argument("--layer-id", type=int, default=3)
    ap.add_argument("--expert-id", type=int, default=18)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    with CacheClient(args.host, args.port) as client:
        resp = client.borrow_expert(
            layer_id=args.layer_id,
            expert_id=args.expert_id,
            device_id=args.device_id,
        )
        assert resp["ok"], resp

        lease_id = resp["lease_id"]
        tensors = resp["tensors"]

        w_up = open_borrowed_tensor(tensors["w_up"], args.device_id)
        w_gate = open_borrowed_tensor(tensors["w_gate"], args.device_id)
        w_down = open_borrowed_tensor(tensors["w_down"], args.device_id)

        try:
            print("lease_id:", lease_id)

            print(
                "w_up shape:", w_up.array.shape,
                "dtype:", w_up.array.dtype,
                "sum_fp32:", float(w_up.array.astype(cp.float32).sum().item()),
                "mean_fp32:", float(w_up.array.astype(cp.float32).mean().item()),
            )
            print(
                "w_gate shape:", w_gate.array.shape,
                "dtype:", w_gate.array.dtype,
                "sum_fp32:", float(w_gate.array.astype(cp.float32).sum().item()),
                "mean_fp32:", float(w_gate.array.astype(cp.float32).mean().item()),
            )
            print(
                "w_down shape:", w_down.array.shape,
                "dtype:", w_down.array.dtype,
                "sum_fp32:", float(w_down.array.astype(cp.float32).sum().item()),
                "mean_fp32:", float(w_down.array.astype(cp.float32).mean().item()),
            )

            hidden = int(w_up.array.shape[1])
            batch_size = int(args.batch_size)

            with cp.cuda.Device(args.device_id):
                x = cp.random.randn(batch_size, hidden, dtype=cp.float32)
                x_fp = x.astype(w_up.array.dtype)

                gate_linear = x_fp @ w_gate.array.T
                up_linear = x_fp @ w_up.array.T
                act = silu(gate_linear.astype(cp.float32)).astype(x_fp.dtype)
                mul = act * up_linear
                out = mul @ w_down.array.T

                print(
                    "input shape:", x_fp.shape,
                    "dtype:", x_fp.dtype,
                    "sum_fp32:", float(x_fp.astype(cp.float32).sum().item()),
                )
                print(
                    "gate_linear shape:", gate_linear.shape,
                    "dtype:", gate_linear.dtype,
                    "sum_fp32:", float(gate_linear.astype(cp.float32).sum().item()),
                    "mean_fp32:", float(gate_linear.astype(cp.float32).mean().item()),
                )
                print(
                    "up_linear shape:", up_linear.shape,
                    "dtype:", up_linear.dtype,
                    "sum_fp32:", float(up_linear.astype(cp.float32).sum().item()),
                    "mean_fp32:", float(up_linear.astype(cp.float32).mean().item()),
                )
                print(
                    "act shape:", act.shape,
                    "dtype:", act.dtype,
                    "sum_fp32:", float(act.astype(cp.float32).sum().item()),
                    "mean_fp32:", float(act.astype(cp.float32).mean().item()),
                )
                print(
                    "mul shape:", mul.shape,
                    "dtype:", mul.dtype,
                    "sum_fp32:", float(mul.astype(cp.float32).sum().item()),
                    "mean_fp32:", float(mul.astype(cp.float32).mean().item()),
                )
                print(
                    "out shape:", out.shape,
                    "dtype:", out.dtype,
                    "sum_fp32:", float(out.astype(cp.float32).sum().item()),
                    "mean_fp32:", float(out.astype(cp.float32).mean().item()),
                )

        finally:
            w_up.close()
            w_gate.close()
            w_down.close()
            print(client.return_expert(lease_id))


if __name__ == "__main__":
    main()
