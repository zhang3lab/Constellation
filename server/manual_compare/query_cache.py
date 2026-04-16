from __future__ import annotations

import argparse
import json

from server.manual_compare.cache_client import CacheClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47000)
    args = ap.parse_args()

    with CacheClient(args.host, args.port) as client:
        resp = client.query()

    if not resp.get("ok", False):
        raise RuntimeError(f"query failed: {resp}")

    cpu_experts = resp.get("cpu_experts", [])
    gpu_experts = resp.get("gpu_experts", [])

    print(f"cpu_resident_count = {len(cpu_experts)}")
    print(f"gpu_resident_count = {len(gpu_experts)}")

    if cpu_experts:
        print("\n[cpu_residents]")
        for item in cpu_experts:
            print(
                f"layer={item['layer_id']} "
                f"expert={item['expert_id']}"
            )

    if gpu_experts:
        print("\n[gpu_residents]")
        for item in gpu_experts:
            print(
                f"layer={item['layer_id']} "
                f"expert={item['expert_id']} "
                f"device={item['device_id']} "
                f"pin_count={item['pin_count']} "
                f"lease_count={item['lease_count']}"
            )

    print("\n[raw_json]")
    print(json.dumps(resp, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
