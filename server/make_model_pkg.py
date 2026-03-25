#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path


def safe_unlink(path: Path):
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        pass


def link_tree(src_root: Path, dst_root: Path):
    for root, dirs, files in os.walk(src_root):
        root_path = Path(root)
        rel = root_path.relative_to(src_root)
        dst_dir = dst_root / rel
        dst_dir.mkdir(parents=True, exist_ok=True)

        init_py = dst_dir / "__init__.py"
        init_py.touch(exist_ok=True)

        for name in files:
            src_file = root_path / name
            dst_file = dst_dir / name

            if dst_file.exists() or dst_file.is_symlink():
                safe_unlink(dst_file)

            os.symlink(src_file, dst_file)

        for name in dirs:
            subdir = dst_dir / name
            subdir.mkdir(parents=True, exist_ok=True)
            (subdir / "__init__.py").touch(exist_ok=True)


def main():
    ap = argparse.ArgumentParser(
        description="Create a writable package shell for a read-only model directory by deep-symlinking files."
    )
    ap.add_argument("model_dir", help="source model directory (read-only is fine)")
    ap.add_argument("tmp_dir", help="destination writable temp directory; used directly as package root")
    ap.add_argument(
        "--package-name",
        default=None,
        help="leaf package name under tmp_dir; defaults to sanitized basename(model_dir)",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="remove existing generated package directory first",
    )
    args = ap.parse_args()

    src_root = Path(args.model_dir).resolve()
    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"source model_dir does not exist or is not a directory: {src_root}")

    tmp_root = Path(args.tmp_dir).resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)

    package_name = args.package_name
    if package_name is None:
        package_name = src_root.name.replace("-", "_").replace(".", "_")

    leaf_root = tmp_root / package_name

    if args.clean and leaf_root.exists():
        shutil.rmtree(leaf_root)

    tmp_root.mkdir(parents=True, exist_ok=True)
    (tmp_root / "__init__.py").touch(exist_ok=True)

    leaf_root.mkdir(parents=True, exist_ok=True)
    (leaf_root / "__init__.py").touch(exist_ok=True)

    link_tree(src_root, leaf_root)

    print("Created package shell:")
    print(f"  source      : {src_root}")
    print(f"  package root: {tmp_root}")
    print(f"  package leaf: {leaf_root}")
    print()
    print("Suggested import path:")
    print(f"  {package_name}")
    print()
    print("Suggested test:")
    print(f"""python3 - <<'PY'
import sys
sys.path.insert(0, "{tmp_root}")
import {package_name}.configuration_deepseek as c
import {package_name}.modeling_deepseek as m
print("ok", c.DeepseekV3Config, m.DeepseekV3Model)
PY""")


if __name__ == "__main__":
    main()
