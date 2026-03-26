#!/usr/bin/env python3
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# =============================================================================
# Config
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent          # .../expert_node_v2
REPO_ROOT = THIS_DIR.parent                        # .../Constellation
BUILD_DIR = THIS_DIR / "build"

CXX = os.environ.get("CXX", "g++")
NVCC = os.environ.get("NVCC", "nvcc")

ENABLE_CUDA = True
ENABLE_AMD = False
ENABLE_INTEL = False

ENABLE_BF16 = True
ENABLE_CUDA_BF16 = True

DEBUG = False
OPT = "-O3"
CXX_STD = "c++20"

# main / test_gpu_info_cuda_v2 / test_down_cuda_v2 / test_fused_up_gate_cuda_v2 /
# test_benchmark_run_expert_cuda_v2
TARGET = "main"

FALLBACK_SMS = ["89"]

# =============================================================================
# Source lists
# =============================================================================

CORE_CPP = [
    "main.cc",
    "control.cc",
    "worker.cc",
    "backend_workspace_v2.cc",
    "node_info.cc",
    "expert_registry_v2.cc",
    "expert_loader_v2.cc",
    "expert_tensor_store_v2.cc",
]

CUDA_CPP = [
    "cuda/backend_workspace_cuda_v2.cc",
    "cuda/backend_cuda_v2.cc",
    "cuda/gpu_info_cuda_v2.cc",
]

CUDA_CU = [
    "cuda/down_cuda_v2.cu",
    "cuda/fused_up_gate_cuda_v2.cu",
    "cuda/fp8_decode_lut_v2.cu",
]

AMD_CPP = [
    "amd/gpu_info_amd_v2.cc",
]

INTEL_CPP = [
    "intel/gpu_info_intel_v2.cc",
]

COMMON_CPP = [
    "../common/protocol.cc",
    "../common/header_codec.cc",
    "../common/infer_codec.cc",
    "../common/inventory_codec.cc",
    "../common/placement_codec.cc",
    "../common/weight_codec.cc",
    "../common/socket_utils.cc",
]

TEST_TARGETS = {
    "test_gpu_info_cuda_v2": {
        "cpp": [
            "cuda/test_gpu_info_cuda_v2.cc",
            "cuda/gpu_info_cuda_v2.cc",
        ],
        "cu": [],
    },
    "test_down_cuda_v2": {
        "cpp": [
            "cuda/test_down_cuda_v2.cc",
        ],
        "cu": [
            "cuda/down_cuda_v2.cu",
            "cuda/fp8_decode_lut_v2.cu",
        ],
    },
    "test_fused_up_gate_cuda_v2": {
        "cpp": [
            "cuda/test_fused_up_gate_cuda_v2.cc",
        ],
        "cu": [
            "cuda/fused_up_gate_cuda_v2.cu",
            "cuda/fp8_decode_lut_v2.cu",
        ],
    },
    "test_benchmark_run_expert_cuda_v2": {
        "cpp": [
            "cuda/test_benchmark_run_expert_cuda_v2.cc",
            "cuda/backend_cuda_v2.cc",
        ],
        "cu": [
            "cuda/down_cuda_v2.cu",
            "cuda/fused_up_gate_cuda_v2.cu",
            "cuda/fp8_decode_lut_v2.cu",
        ],
    },
}

# =============================================================================
# Helpers
# =============================================================================

def quote_cmd(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)

def run(cmd, cwd=None):
    print("+", quote_cmd(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def capture(cmd, cwd=None):
    return subprocess.check_output(cmd, cwd=cwd, text=True)

def detect_sms():
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    queries = [
        [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
        [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
    ]

    sms = []
    for q in queries:
        try:
            out = capture(q).strip()
        except Exception:
            continue
        if not out:
            continue
        for line in out.splitlines():
            m = re.match(r"^\s*(\d+)\.(\d+)\s*$", line.strip())
            if m:
                sms.append(f"{m.group(1)}{m.group(2)}")
        if sms:
            break

    return sorted(set(sms), key=lambda x: int(x))

def gencode_flags(sms):
    flags = []
    for sm in sms:
        flags += ["-gencode", f"arch=compute_{sm},code=sm_{sm}"]
    return flags

def resolve_src(src_rel: str) -> Path:
    return (THIS_DIR / src_rel).resolve()

def obj_path(src_rel: str) -> Path:
    safe = src_rel.replace("../", "__PARENT__/").replace("/", "__")
    return BUILD_DIR / f"{safe}.o"

def existing_sources(srcs):
    out = []
    for s in srcs:
        p = resolve_src(s)
        if p.exists():
            out.append(s)
        else:
            print(f"warning: skip missing source: {s}")
    return out

def common_defines():
    defs = [
        f"-DEXPERT_NODE_V2_ENABLE_CUDA={1 if ENABLE_CUDA else 0}",
        f"-DEXPERT_NODE_V2_ENABLE_AMD={1 if ENABLE_AMD else 0}",
        f"-DEXPERT_NODE_V2_ENABLE_INTEL={1 if ENABLE_INTEL else 0}",
        f"-DEXPERT_NODE_V2_ENABLE_BF16={1 if ENABLE_BF16 else 0}",
        f"-DEXPERT_NODE_V2_ENABLE_CUDA_BF16={1 if ENABLE_CUDA_BF16 else 0}",
    ]
    if DEBUG:
        defs += ["-g", "-DDEBUG=1"]
    return defs

def include_flags():
    return [
        "-I", str(THIS_DIR),
        "-I", str(REPO_ROOT),
    ]

def compile_cpp(src_rel: str):
    src = resolve_src(src_rel)
    obj = obj_path(src_rel)
    obj.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        CXX,
        f"-std={CXX_STD}",
        OPT,
        "-c",
        str(src),
        "-o",
        str(obj),
    ]
    cmd += common_defines()
    cmd += include_flags()

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        cmd += ["-I", str(Path(cuda_home) / "include")]

    run(cmd)
    return obj

def compile_cu(src_rel: str, sms):
    src = resolve_src(src_rel)
    obj = obj_path(src_rel)
    obj.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        NVCC,
        f"-std={CXX_STD}",
        OPT,
        "-c",
        str(src),
        "-o",
        str(obj),
        "--compiler-options",
        "-fPIC",
    ]
    cmd += common_defines()
    cmd += include_flags()
    cmd += gencode_flags(sms)

    if DEBUG:
        cmd += ["-g", "-G"]

    run(cmd)
    return obj

def link_exe(objs, output_name: str):
    out = BUILD_DIR / output_name
    cmd = [CXX, "-o", str(out)] + [str(x) for x in objs]

    if ENABLE_CUDA:
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home:
            cmd += ["-L", str(Path(cuda_home) / "lib64")]
        cmd += ["-lcudart", "-ldl"]

    cmd += ["-lpthread"]
    run(cmd)
    print(f"\nbuilt: {out}")

# =============================================================================
# Build plans
# =============================================================================

def build_main():
    cpp = []
    cpp += CORE_CPP
    cpp += COMMON_CPP
    if ENABLE_CUDA:
        cpp += CUDA_CPP
    if ENABLE_AMD:
        cpp += AMD_CPP
    if ENABLE_INTEL:
        cpp += INTEL_CPP

    cu = CUDA_CU if ENABLE_CUDA else []

    cpp = existing_sources(cpp)
    cu = existing_sources(cu)

    sms = detect_sms() if ENABLE_CUDA else []
    if ENABLE_CUDA and not sms:
        sms = FALLBACK_SMS[:]
        print(f"warning: nvidia-smi probe failed, using fallback sms={sms}")
    elif ENABLE_CUDA:
        print(f"detected sms={sms}")

    objs = []
    for s in cpp:
        objs.append(compile_cpp(s))
    for s in cu:
        objs.append(compile_cu(s, sms))

    link_exe(objs, "expert_node_v2_main")

def build_test(name: str):
    if name not in TEST_TARGETS:
        raise RuntimeError(f"unknown test target: {name}")

    spec = TEST_TARGETS[name]
    cpp = existing_sources(spec["cpp"])
    cu = existing_sources(spec["cu"])

    sms = detect_sms() if ENABLE_CUDA else []
    if ENABLE_CUDA and not sms:
        sms = FALLBACK_SMS[:]
        print(f"warning: nvidia-smi probe failed, using fallback sms={sms}")
    elif ENABLE_CUDA:
        print(f"detected sms={sms}")

    objs = []
    for s in cpp:
        objs.append(compile_cpp(s))
    for s in cu:
        objs.append(compile_cu(s, sms))

    link_exe(objs, name)

# =============================================================================
# Main
# =============================================================================

def main():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    print("=== config ===")
    print(f"THIS_DIR={THIS_DIR}")
    print(f"REPO_ROOT={REPO_ROOT}")
    print(f"BUILD_DIR={BUILD_DIR}")
    print(f"TARGET={TARGET}")
    print(f"ENABLE_CUDA={ENABLE_CUDA}")
    print(f"ENABLE_AMD={ENABLE_AMD}")
    print(f"ENABLE_INTEL={ENABLE_INTEL}")
    print(f"ENABLE_BF16={ENABLE_BF16}")
    print(f"ENABLE_CUDA_BF16={ENABLE_CUDA_BF16}")

    if TARGET == "main":
        build_main()
    else:
        build_test(TARGET)

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"build failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"build failed: {e}", file=sys.stderr)
        sys.exit(1)
