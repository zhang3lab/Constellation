#include <cstdio>
#include <vector>

#include "expert_node_v2/backend/cuda/gpu_info_cuda_v2.h"

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
            common::gpu_vendor_name(gpu.gpu_vendor),
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
            common::gpu_status_name(gpu.gpu_status));
    }

    return 0;
}
