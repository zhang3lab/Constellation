#include <cstdio>
#include <vector>

#include "expert_node_v2/cuda/gpu_info_cuda_v2.h"

static const char* gpu_vendor_name(common::GpuVendor v) {
    switch (v) {
        case common::GpuVendor::Nvidia: return "nvidia";
        case common::GpuVendor::AMD: return "amd";
        case common::GpuVendor::Intel: return "intel";
        default: return "unknown";
    }
}

static const char* gpu_status_name(common::GpuStatus s) {
    switch (s) {
        case common::GpuStatus::Init: return "init";
        case common::GpuStatus::Idle: return "idle";
        case common::GpuStatus::Loading: return "loading";
        case common::GpuStatus::Ready: return "ready";
        case common::GpuStatus::Busy: return "busy";
        case common::GpuStatus::Failed: return "failed";
        default: return "unknown";
    }
}

int main() {
    constexpr int kWorkerIdBegin = 0;
    constexpr int kWorkerPortBase = 50000;

    std::vector<common::StaticGpuInfo> static_gpus;
    if (!BuildLocalCudaGpuInfosV2(
            kWorkerIdBegin,
            kWorkerPortBase,
            &static_gpus)) {
        std::printf("BuildLocalCudaGpuInfosV2 failed\n");
        return 1;
    }

    std::vector<common::DynamicGpuInfo> dynamic_gpus;
    if (!BuildLocalCudaDynamicGpuInfosV2(
            kWorkerIdBegin,
            &dynamic_gpus)) {
        std::printf("BuildLocalCudaDynamicGpuInfosV2 failed\n");
        return 1;
    }

    std::printf("static_gpus=%zu dynamic_gpus=%zu\n",
                static_gpus.size(), dynamic_gpus.size());

    for (const auto& gpu : static_gpus) {
        std::printf(
            "[static] worker_id=%d name=%s vendor=%s total=%llu port=%u flags=0x%x arch=%s\n",
            gpu.worker_id,
            gpu.gpu_name.c_str(),
            gpu_vendor_name(gpu.gpu_vendor),
            static_cast<unsigned long long>(gpu.total_mem_bytes),
            gpu.worker_port,
            gpu.capability_flags,
            gpu.arch_name.c_str());
    }

    for (const auto& gpu : dynamic_gpus) {
        std::printf(
            "[dynamic] worker_id=%d free=%llu status=%s\n",
            gpu.worker_id,
            static_cast<unsigned long long>(gpu.free_mem_bytes),
            gpu_status_name(gpu.gpu_status));
    }

    return 0;
}
