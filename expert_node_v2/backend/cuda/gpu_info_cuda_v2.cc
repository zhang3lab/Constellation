#include "expert_node_v2/backend/cuda/gpu_info_cuda_v2.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "common/protocol.h"

namespace {

std::string make_cuda_arch_name(const cudaDeviceProp& prop) {
    return "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
}

std::uint32_t make_cuda_capability_flags(const cudaDeviceProp& prop) {
    std::uint32_t flags = 0;

    // 对当前 CUDA 路径，现代 NVIDIA GPU 默认视为支持 fp16。
    flags |= common::kGpuCapFp16;

    // Ampere / Ada / Hopper 及以后，先认为支持 bf16。
    if (prop.major >= 8) {
        flags |= common::kGpuCapBf16;
        flags |= common::kGpuCapFp8;   // 先作为 capability hint，后面可细化
    }

    return flags;
}

common::GpuStatus make_initial_gpu_status() {
    return common::GpuStatus::Idle;
}

}  // namespace

bool BuildLocalCudaGpuInfosV2(
    std::int32_t worker_id_begin,
    std::uint32_t worker_port_base,
    std::vector<common::StaticGpuInfo>* out) {
    if (out == nullptr) return false;
    out->clear();

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return false;
    }

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            continue;
        }

        common::StaticGpuInfo gpu;
        gpu.worker_id = worker_id_begin + i;
        gpu.gpu_name = prop.name;
        gpu.total_mem_bytes = static_cast<std::uint64_t>(prop.totalGlobalMem);
        gpu.worker_port =
            worker_port_base + static_cast<std::uint32_t>(gpu.worker_id);

        gpu.gpu_vendor = common::GpuVendor::Nvidia;
        gpu.capability_flags = make_cuda_capability_flags(prop);
        gpu.arch_name = make_cuda_arch_name(prop);

        out->push_back(std::move(gpu));
    }

    return true;
}

bool BuildLocalCudaDynamicGpuInfosV2(
    std::int32_t worker_id_begin,
    std::vector<common::DynamicGpuInfo>* out) {
    if (out == nullptr) return false;
    out->clear();

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return false;
    }

    for (int i = 0; i < count; ++i) {
        std::size_t free_mem = 0;
        std::size_t total_mem = 0;

        err = cudaSetDevice(i);
        if (err != cudaSuccess) {
            continue;
        }

        err = cudaMemGetInfo(&free_mem, &total_mem);
        if (err != cudaSuccess) {
            continue;
        }

        common::DynamicGpuInfo gpu;
        gpu.worker_id = worker_id_begin + i;
        gpu.free_mem_bytes = static_cast<std::uint64_t>(free_mem);
        gpu.gpu_status = make_initial_gpu_status();

        out->push_back(std::move(gpu));
    }

    return true;
}
