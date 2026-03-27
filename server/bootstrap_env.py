#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    ap = argparse.ArgumentParser(
        description="One-shot bootstrap for server env: install flash-attn wheel, install requirements, link model package."
    )
    ap.add_argument(
        "--model-dir",
        default="/model/ModelScope/deepseek-ai/DeepSeek-V3.1",
        help="source model directory",
    )
    ap.add_argument(
        "--tmp-dir",
        default="tmp",
        help="temporary package root",
    )
    ap.add_argument(
        "--package-name",
        default="DeepSeek_V3_1",
        help="generated import package name",
    )
    ap.add_argument(
        "--flash-attn-version",
        default="2.8.3",
        help="flash-attn version in wheel filename",
    )
    ap.add_argument(
        "--flash-wheel-release",
        default="v0.9.0",
        help="GitHub release tag for prebuilt flash-attn wheels",
    )
    ap.add_argument(
        "--flash-wheel-base",
        default="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download",
        help="base URL for flash-attn prebuilt wheels",
    )
    ap.add_argument(
        "--skip-flash-attn",
        action="store_true",
        help="skip flash-attn wheel install",
    )
    ap.add_argument(
        "--skip-requirements",
        action="store_true",
        help="skip pip install -r requirements.txt",
    )
    ap.add_argument(
        "--skip-model-pkg",
        action="store_true",
        help="skip make_model_pkg step",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="clean generated package dir before relinking",
    )
    return ap.parse_args()


def python_tag() -> str:
    v = sys.version_info
    return f"cp{v.major}{v.minor}"


def parse_torch_minor(torch_ver: str) -> str:
    m = re.match(r"^(\d+)\.(\d+)", torch_ver)
    if not m:
        raise RuntimeError(f"cannot parse torch version: {torch_ver}")
    return f"{m.group(1)}.{m.group(2)}"


def parse_cuda_tag(torch_ver: str, torch_cuda: str | None) -> str:
    m = re.search(r"\+cu(\d+)", torch_ver)
    if m:
        return f"cu{m.group(1)}"

    if torch_cuda:
        m = re.match(r"^(\d+)\.(\d+)", torch_cuda)
        if m:
            return f"cu{m.group(1)}{m.group(2)}"

    raise RuntimeError(
        f"cannot determine CUDA tag from torch={torch_ver}, torch.version.cuda={torch_cuda}"
    )


def build_flash_attn_url(
    base: str,
    release: str,
    flash_ver: str,
    torch_minor: str,
    cu_tag: str,
    py_tag: str,
) -> str:
    filename = (
        f"flash_attn-{flash_ver}+{cu_tag}torch{torch_minor}-"
        f"{py_tag}-{py_tag}-linux_x86_64.whl"
    )
    filename_url = filename.replace("+", "%2B")
    return f"{base}/{release}/{filename_url}"


def uninstall_flash_attn():
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "flash-attn", "flash_attn"],
        check=False,
    )


def install_flash_attn(url: str):
    uninstall_flash_attn()
    run([sys.executable, "-m", "pip", "install", url])


def install_requirements():
    repo_root = Path(__file__).resolve().parent.parent
    req = repo_root / "requirements.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(req)])
    else:
        print(f"warning: requirements.txt not found at {req}")


def install_base_tools():
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "pip",
            "setuptools",
            "wheel",
            "packaging",
            "ninja",
            "psutil",
            "safetensors",
        ]
    )


def verify_flash_attn():
    code = (
        "import torch, flash_attn\n"
        "print('torch:', torch.__version__)\n"
        "print('torch cuda:', torch.version.cuda)\n"
        "print('flash_attn:', flash_attn.__file__)\n"
    )
    run([sys.executable, "-c", code])


def make_model_pkg(model_dir: str, tmp_dir: str, package_name: str, clean: bool):
    cmd = [
        sys.executable,
        "-m",
        "server.make_model_pkg",
        model_dir,
        tmp_dir,
        "--package-name",
        package_name,
    ]
    if clean:
        cmd.append("--clean")
    run(cmd)


def verify_model_pkg(tmp_dir: str, package_name: str):
    leaf = Path(tmp_dir).resolve() / package_name
    needed = [
        leaf / "__init__.py",
        leaf / "configuration_deepseek.py",
        leaf / "modeling_deepseek.py",
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise RuntimeError(f"generated package missing files: {missing}")

    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(tmp_dir).resolve())!r})\n"
        f"import {package_name}.configuration_deepseek as c\n"
        f"import {package_name}.modeling_deepseek as m\n"
        "print('ok', c.DeepseekV3Config, m.DeepseekV3Model)\n"
    )
    run([sys.executable, "-c", code])


def main():
    args = parse_args()

    import torch

    torch_ver = torch.__version__
    torch_minor = parse_torch_minor(torch_ver)
    cu_tag = parse_cuda_tag(torch_ver, torch.version.cuda)
    py_tag = python_tag()

    print("=== detected environment ===")
    print("python_tag  =", py_tag)
    print("torch       =", torch_ver)
    print("torch_minor =", torch_minor)
    print("torch.cuda  =", torch.version.cuda)
    print("cuda_tag    =", cu_tag)

    if not args.skip_flash_attn:
        wheel_url = build_flash_attn_url(
            base=args.flash_wheel_base,
            release=args.flash_wheel_release,
            flash_ver=args.flash_attn_version,
            torch_minor=torch_minor,
            cu_tag=cu_tag,
            py_tag=py_tag,
        )
        print("flash_attn wheel url =", wheel_url)
        install_flash_attn(wheel_url)
        verify_flash_attn()

    if not args.skip_requirements:
        install_requirements()

    install_base_tools()

    if not args.skip_model_pkg:
        make_model_pkg(
            model_dir=args.model_dir,
            tmp_dir=args.tmp_dir,
            package_name=args.package_name,
            clean=args.clean,
        )
        verify_model_pkg(args.tmp_dir, args.package_name)

    print("=== bootstrap done ===")


if __name__ == "__main__":
    main()
