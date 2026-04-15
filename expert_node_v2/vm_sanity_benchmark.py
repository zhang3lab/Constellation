import os
import socket
import subprocess
import sys
import time
from typing import Optional


def run(cmd: str) -> str:
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    out = p.stdout.strip()
    err = p.stderr.strip()
    if out and err:
        return out + "\n" + err
    return out or err


def print_section(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def memcopy_benchmark(size_mb: int = 256, rounds: int = 20) -> None:
    n = size_mb * 1024 * 1024
    src = bytearray(n)
    t0 = time.time()
    for _ in range(rounds):
        _ = bytes(src)
    t1 = time.time()
    dt = t1 - t0
    gib_s = (n * rounds) / dt / (1024 ** 3)
    print(f"memcpy_bytes={n * rounds}")
    print(f"seconds={dt:.6f}")
    print(f"GiB_per_s={gib_s:.6f}")


def maybe_run(cmd: str) -> None:
    out = run(f"command -v {cmd}")
    if out:
        print(f"$ {cmd} present: {out.splitlines()[0]}")
    else:
        print(f"$ {cmd} not found")


def main(argv: list[str]) -> int:
    peer: Optional[str] = argv[1] if len(argv) > 1 else None

    print_section("BASIC")
    print(f"hostname={socket.gethostname()}")
    print(f"python={sys.version.split()[0]}")
    print(f"peer={peer or ''}")

    print_section("CPU")
    print(run("nproc"))
    print(run("lscpu | egrep 'Model name|CPU\\(s\\)|Thread|Core|Socket|Hypervisor' || true"))

    print_section("MEMORY")
    print(run("free -h"))
    print(run("cat /proc/meminfo | head -20"))

    print_section("NET")
    print(run("ifconfig -a || true"))
    print(run("route -n || true"))
    print(run("netstat -i || true"))
    print(run("cat /sys/class/net/eth0/operstate 2>/dev/null || true"))
    print(run("cat /sys/class/net/eth0/mtu 2>/dev/null || true"))
    print(run("cat /sys/class/net/eth0/speed 2>/dev/null || true"))
    print(run("cat /sys/class/net/eth0/duplex 2>/dev/null || true"))
    print(run("readlink -f /sys/class/net/eth0/device/driver 2>/dev/null || true"))

    print_section("TOOLS")
    for cmd in ["iperf3", "lscpu", "free", "ifconfig", "route", "netstat"]:
        maybe_run(cmd)

    print_section("MEMCOPY_BENCH")
    memcopy_benchmark()

    if peer:
        print_section("IPERF_HINTS")
        print("Run one side as server:")
        print("  iperf3 -s")
        print("Run from this VM to peer:")
        print(f"  iperf3 -c {peer} -P 4 -t 20")
        print("Run reverse:")
        print(f"  iperf3 -c {peer} -P 4 -t 20 -R")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

