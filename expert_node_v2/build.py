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


def _enabled_backend_specs():
    cache = getattr(_enabled_backend_specs, "_cache", None)
    if cache is not None:
        return cache

    out = {}
    for name, spec in config.BACKENDS.items():
        if spec.get("enabled", False):
            out[name] = spec

    _enabled_backend_specs._cache = out
    return out


def _src_extra_cflags_map():
    cache = getattr(_src_extra_cflags_map, "_cache", None)
    if cache is not None:
        return cache

    out = {}
    for _, spec in _enabled_backend_specs().items():
        cflags = list(spec.get("cflags", []))
        for src in spec.get("src", []):
            norm = src.replace("\\", "/")
            out[norm] = cflags

    _src_extra_cflags_map._cache = out
    return out


def _src_extra_cflags(src_rel: str):
    norm = src_rel.replace("\\", "/")
    return list(_src_extra_cflags_map().get(norm, []))


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
                extra_cflags=_src_extra_cflags(s),
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
                extra_cflags=_src_extra_cflags(s),
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


def _backend_src_prefixes():
    out = {}
    for backend_name, spec in config.BACKENDS.items():
        prefixes = set()
        for src in spec.get("src", []):
            norm = src.replace("\\", "/")
            marker = f"backend/{backend_name}/"
            idx = norm.find(marker)
            if idx >= 0:
                prefixes.add(marker)
        out[backend_name] = sorted(prefixes)
    return out


def _test_required_backends(test_name: str):
    spec = config.TEST_TARGETS[test_name]
    srcs = [s.replace("\\", "/") for s in spec.get("src", [])]

    required = set()
    backend_prefixes = _backend_src_prefixes()

    for backend_name, prefixes in backend_prefixes.items():
        for prefix in prefixes:
            if any(src.startswith(prefix) for src in srcs):
                required.add(backend_name)
                break

    return sorted(required)


def should_run_test(name: str) -> bool:
    required = _test_required_backends(name)
    if not required:
        return True

    for backend_name in required:
        backend_spec = config.BACKENDS.get(backend_name, {})
        if not backend_spec.get("enabled", False):
            return False

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
