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
    "backend/backend_workspace_v2.cc",
    "node_info.cc",
    "expert_registry_v2.cc",
    "expert_format_v2.cc",
]

CUDA_CPP = [
    "backend/cuda/backend_workspace_cuda_v2.cc",
    "backend/cuda/backend_cuda_v2.cc",
    "backend/cuda/gpu_info_cuda_v2.cc",
]

CUDA_CU = [
    "backend/cuda/down_cuda_v2.cu",
    "backend/cuda/fused_up_gate_cuda_v2.cu",
    "backend/cuda/fp8_decode_lut_v2.cu",
]

AMD_CPP = [
    "backend/amd/gpu_info_amd_v2.cc",
]

INTEL_CPP = [
    "backend/intel/gpu_info_intel_v2.cc",
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
            "backend/cuda/tests/test_gpu_info_cuda_v2.cc",
            "backend/cuda/gpu_info_cuda_v2.cc",
        ],
        "cu": [],
    },
    "test_down_cuda_v2": {
        "cpp": [
            "backend/cuda/tests/test_down_cuda_v2.cc",
        ],
        "cu": [
            "backend/cuda/down_cuda_v2.cu",
            "backend/cuda/fp8_decode_lut_v2.cu",
        ],
    },
    "test_fused_up_gate_cuda_v2": {
        "cpp": [
            "backend/cuda/tests/test_fused_up_gate_cuda_v2.cc",
        ],
        "cu": [
            "backend/cuda/fused_up_gate_cuda_v2.cu",
            "backend/cuda/fp8_decode_lut_v2.cu",
        ],
    },
    "test_benchmark_run_expert_cuda_v2": {
        "cpp": [
            "backend/cuda/tests/test_benchmark_run_expert_cuda_v2.cc",
            "backend/cuda/backend_cuda_v2.cc",
            "expert_format_v2.cc",
        ],
        "cu": [
            "backend/cuda/down_cuda_v2.cu",
            "backend/cuda/fused_up_gate_cuda_v2.cu",
            "backend/cuda/fp8_decode_lut_v2.cu",
        ],
    },
}
