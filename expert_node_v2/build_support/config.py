from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = THIS_DIR.parent
BUILD_DIR = THIS_DIR / "build"

CXX = "g++"
NVCC = "nvcc"

ENABLE_CUDA = True
ENABLE_AMD = False
ENABLE_INTEL = False

ENABLE_BF16 = True
ENABLE_CUDA_BF16 = True

DEFAULT_DEBUG = False
OPT = "-O3"
CXX_STD = "c++20"

CORE_CPP = [
    "main.cc",
    "control.cc",
    "worker.cc",
    "backend_workspace_v2.cc",
    "node_info.cc",
    "expert_registry_v2.cc",
    "expert_format_v2.cc",
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
