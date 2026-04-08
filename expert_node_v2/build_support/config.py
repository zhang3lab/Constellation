from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = THIS_DIR.parent
BUILD_DIR = THIS_DIR / "build"

CXX = "g++"
NVCC = "nvcc"

ENABLE_CPU = True
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
    "backend/fp8_lut_v2.cc",
    "backend/expert_reference_v2.cc",
    "backend/dummy_expert_data_v2.cc",
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

# Source kind is inferred from file suffix for now.
# If a future backend needs files whose toolchain cannot be determined by suffix
# alone, extend the source spec format (for example, {"path": ..., "kind": ...})
# and update toolchain.resolve_source_kind(...) accordingly.
SOURCE_RULES = {
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cu": "cuda",
}

TOOLCHAINS = {
    "cpp": {
        "compiler": CXX,
    },
    "cuda": {
        "compiler": NVCC,
    },
}

BACKENDS = {
    "cpu": {
        "enabled": ENABLE_CPU,
        "src": [
            "backend/cpu/backend_workspace_cpu_v2.cc",
            "backend/cpu/backend_cpu_v2.cc",
            "backend/cpu/fused_up_gate_cpu_v2.cc",
            "backend/cpu/down_cpu_v2.cc",
        ],
    },
    "cuda": {
        "enabled": ENABLE_CUDA,
        "src": [
            "backend/cuda/backend_workspace_cuda_v2.cc",
            "backend/cuda/backend_cuda_v2.cc",
            "backend/cuda/gpu_info_cuda_v2.cc",
            "backend/cuda/down_cuda_v2.cu",
            "backend/cuda/fused_up_gate_cuda_v2.cu",
            "backend/cuda/fp8_decode_lut_v2.cu",
        ],
    },
    "amd": {
        "enabled": ENABLE_AMD,
        "src": [
            "backend/amd/gpu_info_amd_v2.cc",
        ],
    },
    "intel": {
        "enabled": ENABLE_INTEL,
        "src": [
            "backend/intel/gpu_info_intel_v2.cc",
        ],
    },
}

FEATURE_DEFINES = {
    "EXPERT_NODE_V2_ENABLE_CPU": ENABLE_CPU,
    "EXPERT_NODE_V2_ENABLE_CUDA": ENABLE_CUDA,
    "EXPERT_NODE_V2_ENABLE_AMD": ENABLE_AMD,
    "EXPERT_NODE_V2_ENABLE_INTEL": ENABLE_INTEL,
    "EXPERT_NODE_V2_ENABLE_BF16": ENABLE_BF16,
    "EXPERT_NODE_V2_ENABLE_CUDA_BF16": ENABLE_CUDA_BF16,
}

TEST_TARGETS = {
    "test_activation_codec_v2": {
        "src": [
            "backend/tests/test_activation_codec_v2.cc",
        ],
    },
    "test_activation_codec_cuda_v2": {
        "src": [
            "backend/cuda/tests/test_activation_codec_cuda_v2.cc",
        ],
    },
    "test_gpu_info_cuda_v2": {
        "src": [
            "backend/cuda/tests/test_gpu_info_cuda_v2.cc",
            "backend/cuda/gpu_info_cuda_v2.cc",
        ],
    },
    "test_down_cuda_v2": {
        "src": [
            "backend/cuda/tests/test_down_cuda_v2.cc",
            "backend/cuda/backend_cuda_v2.cc",
            "expert_format_v2.cc",
            "backend/expert_reference_v2.cc",
            "backend/dummy_expert_data_v2.cc",
            "backend/fp8_lut_v2.cc",
            "backend/cuda/down_cuda_v2.cu",
            "backend/cuda/fused_up_gate_cuda_v2.cu",
            "backend/cuda/fp8_decode_lut_v2.cu",
        ],
    },
    "test_fused_up_gate_cuda_v2": {
        "src": [
            "backend/cuda/tests/test_fused_up_gate_cuda_v2.cc",
            "backend/cuda/backend_cuda_v2.cc",
            "expert_format_v2.cc",
            "backend/expert_reference_v2.cc",
            "backend/dummy_expert_data_v2.cc",
            "backend/fp8_lut_v2.cc",
            "backend/cuda/down_cuda_v2.cu",
            "backend/cuda/fused_up_gate_cuda_v2.cu",
            "backend/cuda/fp8_decode_lut_v2.cu",
        ],
    },
    "test_benchmark_run_expert_cuda_v2": {
        "src": [
            "backend/cuda/tests/test_benchmark_run_expert_cuda_v2.cc",
            "backend/cuda/backend_cuda_v2.cc",
            "expert_format_v2.cc",
            "backend/expert_reference_v2.cc",
            "backend/dummy_expert_data_v2.cc",
            "backend/fp8_lut_v2.cc",
            "backend/cuda/down_cuda_v2.cu",
            "backend/cuda/fused_up_gate_cuda_v2.cu",
            "backend/cuda/fp8_decode_lut_v2.cu",
        ],
    },
    "test_fused_up_gate_cpu_v2": {
        "src": [
            "backend/cpu/tests/test_fused_up_gate_cpu_v2.cc",
            "backend/cpu/fused_up_gate_cpu_v2.cc",
            "expert_format_v2.cc",
            "backend/expert_reference_v2.cc",
            "backend/dummy_expert_data_v2.cc",
            "backend/fp8_lut_v2.cc",
        ],
    },
    "test_down_cpu_v2": {
        "src": [
            "backend/cpu/tests/test_down_cpu_v2.cc",
            "backend/cpu/down_cpu_v2.cc",
            "expert_format_v2.cc",
            "backend/expert_reference_v2.cc",
            "backend/dummy_expert_data_v2.cc",
            "backend/fp8_lut_v2.cc",
        ],
    },
    "test_benchmark_run_expert_cpu_v2": {
        "src": [
            "backend/cpu/tests/test_benchmark_run_expert_cpu_v2.cc",
            "backend/cpu/backend_cpu_v2.cc",
            "backend/cpu/fused_up_gate_cpu_v2.cc",
            "backend/cpu/down_cpu_v2.cc",
            "expert_format_v2.cc",
            "backend/expert_reference_v2.cc",
            "backend/dummy_expert_data_v2.cc",
            "backend/fp8_lut_v2.cc",
        ],
    },
}
