#!/usr/bin/env python3
import argparse
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
        enable_cpu=config.ENABLE_CPU,
        enable_cuda=config.ENABLE_CUDA,
        enable_amd=config.ENABLE_AMD,
        enable_intel=config.ENABLE_INTEL,
        enable_bf16=config.ENABLE_BF16,
        enable_cuda_bf16=config.ENABLE_CUDA_BF16,
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
        enable_cpu=config.ENABLE_CPU,
        enable_cuda=config.ENABLE_CUDA,
        enable_amd=config.ENABLE_AMD,
        enable_intel=config.ENABLE_INTEL,
        enable_bf16=config.ENABLE_BF16,
        enable_cuda_bf16=config.ENABLE_CUDA_BF16,
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

    for backend_name, backend_spec in config.BACKENDS.items():
        print(f"ENABLE_{backend_name.upper()}={backend_spec.get('enabled', False)}")

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
