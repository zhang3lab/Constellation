from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_CONFIG = str(TEST_DIR / "config.json")


def list_tests() -> list[Path]:
    files = []
    for p in sorted(TEST_DIR.glob("*.py")):
        if p.name in {"run.py", "utils.py", "__init__.py"}:
            continue
        files.append(p)
    return files


def select_tests(kind: str, pattern: str | None) -> list[Path]:
    tests = list_tests()

    if kind == "smoke":
        tests = [p for p in tests if p.name.startswith("1_")]
    elif kind == "regress":
        tests = [p for p in tests if p.name.startswith("2_")]
    elif kind == "e2e":
        tests = [p for p in tests if p.name.startswith("3_")]
    elif kind == "all":
        pass
    else:
        raise RuntimeError(f"unknown kind: {kind}")

    if pattern:
        tests = [p for p in tests if pattern in p.name]

    return tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kind",
        type=str,
        default="all",
        choices=["smoke", "regress", "e2e", "all"],
    )
    ap.add_argument("--pattern", type=str, default=None)
    ap.add_argument("--config", type=str, default=DEFAULT_TEST_CONFIG)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    tests = select_tests(args.kind, args.pattern)

    if args.list:
        for p in tests:
            print(p.name)
        return

    if not tests:
        raise RuntimeError("no tests selected")

    print(f"[test-runner] config={args.config}")

    for p in tests:
        print(f"=== running {p.name} ===")
        old_argv = sys.argv[:]
        try:
            sys.argv = [str(p), "--config", args.config]
            runpy.run_path(str(p), run_name="__main__")
        finally:
            sys.argv = old_argv
        print(f"=== passed  {p.name} ===")


if __name__ == "__main__":
    main()
