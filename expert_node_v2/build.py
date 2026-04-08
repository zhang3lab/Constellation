#!/usr/bin/env python3
import argparse
import subprocess
import sys

from build_support import config, toolchain


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="main",
        help="build target name, default: main",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="build with debug flags",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove build directory before building",
    )
    return parser.parse_args()


def enabled_backend_src():
    out = []
    for _, spec in config.BACKENDS.items():
        if spec.get("enabled", False):
            out += spec.get("src", [])
    return out


def build_main(debug: bool):
    src = []
    src += config.CORE_CPP
    src += config.COMMON_CPP
    src += enabled_backend_src()

    src = toolchain.existing_sources(config.THIS_DIR, src)

    defines = toolchain.common_defines(
        feature_defines=config.FEATURE_DEFINES,
        debug=debug,
    )

    objs = []
    for s in src:
        objs.append(
            toolchain.compile_source(
                project_root=config.THIS_DIR,
                repo_root=config.REPO_ROOT,
                build_dir=config.BUILD_DIR,
                src_rel=s,
                cxx_std=config.CXX_STD,
                opt=config.OPT,
                defines=defines,
                debug=debug,
                enable_cuda=config.ENABLE_CUDA,
                source_rules=config.SOURCE_RULES,
                toolchains=config.TOOLCHAINS,
            )
        )

    out = config.BUILD_DIR / "expert_node_v2_main"
    toolchain.link_exe(
        cxx=config.CXX,
        output_path=out,
        objs=objs,
        enable_cuda=config.ENABLE_CUDA,
    )
    print(f"\nbuilt: {out}")


def build_test(name: str, debug: bool):
    if name not in config.TEST_TARGETS:
        raise RuntimeError(f"unknown test target: {name}")

    spec = config.TEST_TARGETS[name]
    src = toolchain.existing_sources(config.THIS_DIR, spec["src"])

    defines = toolchain.common_defines(
        feature_defines=config.FEATURE_DEFINES,
        debug=debug,
    )

    objs = []
    for s in src:
        objs.append(
            toolchain.compile_source(
                project_root=config.THIS_DIR,
                repo_root=config.REPO_ROOT,
                build_dir=config.BUILD_DIR,
                src_rel=s,
                cxx_std=config.CXX_STD,
                opt=config.OPT,
                defines=defines,
                debug=debug,
                enable_cuda=config.ENABLE_CUDA,
                source_rules=config.SOURCE_RULES,
                toolchains=config.TOOLCHAINS,
            )
        )

    out = config.BUILD_DIR / name
    toolchain.link_exe(
        cxx=config.CXX,
        output_path=out,
        objs=objs,
        enable_cuda=config.ENABLE_CUDA,
    )
    print(f"\nbuilt: {out}")


def _enabled_backend_test_prefixes():
    prefixes = []

    if config.ENABLE_CPU:
        prefixes.append("test_")
        prefixes.append("test_activation_codec_v2")

    if config.ENABLE_CUDA:
        prefixes.append("test_gpu_info_cuda_v2")
        prefixes.append("test_activation_codec_cuda_v2")

    return prefixes


def should_run_test(name: str) -> bool:
    # codec cpu test is backend-agnostic enough to run always
    if name == "test_activation_codec_v2":
        return True

    if "_cpu_" in name:
        return config.ENABLE_CPU
    if "_cuda_" in name:
        return config.ENABLE_CUDA
    if "_amd_" in name:
        return config.ENABLE_AMD
    if "_intel_" in name:
        return config.ENABLE_INTEL

    # fallback: run unknown/general tests
    return True


def run_test_binary(name: str):
    exe = config.BUILD_DIR / name
    if not exe.exists():
        raise RuntimeError(f"test binary not found: {exe}")

    cmd = [str(exe)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_regression(debug: bool):
    selected = [name for name in config.TEST_TARGETS.keys() if should_run_test(name)]

    if not selected:
        print("no regression tests selected")
        return

    print("=== regression tests ===")
    for name in selected:
        print(name)

    for name in selected:
        print(f"\n=== build {name} ===")
        build_test(name=name, debug=debug)

    for name in selected:
        print(f"\n=== run {name} ===")
        run_test_binary(name)

    print("\nregression: PASS")


def print_config(target: str, debug: bool):
    print("=== config ===")
    print(f"THIS_DIR={config.THIS_DIR}")
    print(f"REPO_ROOT={config.REPO_ROOT}")
    print(f"BUILD_DIR={config.BUILD_DIR}")
    print(f"TARGET={target}")
    print(f"DEBUG={debug}")

    for name, enabled in config.FEATURE_DEFINES.items():
        print(f"{name}={enabled}")


def main():
    args = parse_args()
    target = args.target
    debug = args.debug or config.DEFAULT_DEBUG

    if args.clean:
        toolchain.clean_dir(config.BUILD_DIR)

    config.BUILD_DIR.mkdir(parents=True, exist_ok=True)
    print_config(target=target, debug=debug)

    if target == "main":
        build_main(debug=debug)
    elif target == "regression":
        build_regression(debug=debug)
    else:
        build_test(target, debug=debug)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"build failed: {e}", file=sys.stderr)
        sys.exit(1)
