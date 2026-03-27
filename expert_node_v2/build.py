#!/usr/bin/env python3
import argparse
import sys

from build_support import amd_backend, config, cuda_backend, intel_backend, toolchain


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


def build_main(debug: bool):
    cpp = []
    cpp += config.CORE_CPP
    cpp += config.COMMON_CPP
    if config.ENABLE_CUDA:
        cpp += config.CUDA_CPP
    if config.ENABLE_AMD:
        cpp += config.AMD_CPP
    if config.ENABLE_INTEL:
        cpp += config.INTEL_CPP

    cu = config.CUDA_CU if config.ENABLE_CUDA else []

    cpp = toolchain.existing_sources(config.THIS_DIR, cpp)
    cu = toolchain.existing_sources(config.THIS_DIR, cu)

    defines = toolchain.common_defines(
        enable_cuda=config.ENABLE_CUDA,
        enable_amd=config.ENABLE_AMD,
        enable_intel=config.ENABLE_INTEL,
        enable_bf16=config.ENABLE_BF16,
        enable_cuda_bf16=config.ENABLE_CUDA_BF16,
        debug=debug,
    )

    objs = []
    for s in cpp:
        objs.append(
            toolchain.compile_cpp(
                cxx=config.CXX,
                project_root=config.THIS_DIR,
                repo_root=config.REPO_ROOT,
                build_dir=config.BUILD_DIR,
                src_rel=s,
                cxx_std=config.CXX_STD,
                opt=config.OPT,
                defines=defines,
                debug=debug,
            )
        )

    if config.ENABLE_CUDA:
        objs += cuda_backend.build_objects(
            nvcc=config.NVCC,
            project_root=config.THIS_DIR,
            repo_root=config.REPO_ROOT,
            build_dir=config.BUILD_DIR,
            sources=cu,
            cxx_std=config.CXX_STD,
            opt=config.OPT,
            defines=defines,
            debug=debug,
        )

    if config.ENABLE_AMD:
        objs += amd_backend.build_objects()

    if config.ENABLE_INTEL:
        objs += intel_backend.build_objects()

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
    cpp = toolchain.existing_sources(config.THIS_DIR, spec["cpp"])
    cu = toolchain.existing_sources(config.THIS_DIR, spec["cu"])

    defines = toolchain.common_defines(
        enable_cuda=config.ENABLE_CUDA,
        enable_amd=config.ENABLE_AMD,
        enable_intel=config.ENABLE_INTEL,
        enable_bf16=config.ENABLE_BF16,
        enable_cuda_bf16=config.ENABLE_CUDA_BF16,
        debug=debug,
    )

    objs = []
    for s in cpp:
        objs.append(
            toolchain.compile_cpp(
                cxx=config.CXX,
                project_root=config.THIS_DIR,
                repo_root=config.REPO_ROOT,
                build_dir=config.BUILD_DIR,
                src_rel=s,
                cxx_std=config.CXX_STD,
                opt=config.OPT,
                defines=defines,
                debug=debug,
            )
        )

    if config.ENABLE_CUDA and cu:
        objs += cuda_backend.build_objects(
            nvcc=config.NVCC,
            project_root=config.THIS_DIR,
            repo_root=config.REPO_ROOT,
            build_dir=config.BUILD_DIR,
            sources=cu,
            cxx_std=config.CXX_STD,
            opt=config.OPT,
            defines=defines,
            debug=debug,
        )

    out = config.BUILD_DIR / name
    toolchain.link_exe(
        cxx=config.CXX,
        output_path=out,
        objs=objs,
        enable_cuda=config.ENABLE_CUDA,
    )
    print(f"\nbuilt: {out}")


def main():
    args = parse_args()
    target = args.target
    debug = args.debug or config.DEFAULT_DEBUG

    if args.clean:
        toolchain.clean_dir(config.BUILD_DIR)

    config.BUILD_DIR.mkdir(parents=True, exist_ok=True)

    print("=== config ===")
    print(f"THIS_DIR={config.THIS_DIR}")
    print(f"REPO_ROOT={config.REPO_ROOT}")
    print(f"BUILD_DIR={config.BUILD_DIR}")
    print(f"TARGET={target}")
    print(f"DEBUG={debug}")
    print(f"ENABLE_CUDA={config.ENABLE_CUDA}")
    print(f"ENABLE_AMD={config.ENABLE_AMD}")
    print(f"ENABLE_INTEL={config.ENABLE_INTEL}")
    print(f"ENABLE_BF16={config.ENABLE_BF16}")
    print(f"ENABLE_CUDA_BF16={config.ENABLE_CUDA_BF16}")

    if target == "main":
        build_main(debug=debug)
    else:
        build_test(target, debug=debug)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"build failed: {e}", file=sys.stderr)
        sys.exit(1)
