#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None, check=True):
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=check, cwd=cwd)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Bootstrap server env: init submodules, apply third_party patches, install requirements and base tools."
    )
    ap.add_argument(
        "--skip-submodules",
        action="store_true",
        help="skip git submodule update --init --recursive",
    )
    ap.add_argument(
        "--skip-third-party-patches",
        action="store_true",
        help="skip applying patches under third_party/patches",
    )
    ap.add_argument(
        "--skip-requirements",
        action="store_true",
        help="skip pip install -r requirements.txt",
    )
    ap.add_argument(
        "--force-reapply-patches",
        action="store_true",
        help="restore third_party submodules before reapplying patches",
    )
    return ap.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def install_requirements():
    req = repo_root() / "requirements.txt"
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


def init_submodules():
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_root())


def git_apply_check(submodule_dir: Path, patch_path: Path) -> bool:
    r = subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=submodule_dir,
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def git_apply_reverse_check(submodule_dir: Path, patch_path: Path) -> bool:
    r = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch_path)],
        cwd=submodule_dir,
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def apply_patch_to_submodule(submodule_dir: Path, patch_path: Path, force_reapply: bool):
    if force_reapply:
        run(["git", "restore", "."], cwd=submodule_dir)
        run(["git", "clean", "-fd"], cwd=submodule_dir)

    if git_apply_check(submodule_dir, patch_path):
        run(["git", "apply", str(patch_path)], cwd=submodule_dir)
        return

    if git_apply_reverse_check(submodule_dir, patch_path):
        print(f"patch already applied: {patch_path}")
        return

    raise RuntimeError(
        f"cannot apply patch cleanly and patch does not appear to be already applied: {patch_path}"
    )


def apply_third_party_patches(force_reapply: bool):
    root = repo_root()
    patches_root = root / "third_party" / "patches"
    if not patches_root.exists():
        print(f"warning: no patch directory found at {patches_root}")
        return

    for component_dir in sorted(patches_root.iterdir()):
        if not component_dir.is_dir():
            continue

        submodule_dir = root / "third_party" / component_dir.name
        if not submodule_dir.exists():
            raise RuntimeError(
                f"patch directory exists but submodule dir does not: {submodule_dir}"
            )

        patch_files = sorted(component_dir.glob("*.patch"))
        for patch_file in patch_files:
            apply_patch_to_submodule(
                submodule_dir=submodule_dir,
                patch_path=patch_file.resolve(),
                force_reapply=force_reapply,
            )


def main():
    args = parse_args()

    print("=== bootstrap start ===")
    print("repo_root =", repo_root())

    if not args.skip_submodules:
        init_submodules()

    if not args.skip_third_party_patches:
        apply_third_party_patches(force_reapply=args.force_reapply_patches)

    if not args.skip_requirements:
        install_requirements()

    install_base_tools()

    print("=== bootstrap done ===")


if __name__ == "__main__":
    main()
