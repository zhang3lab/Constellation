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

int main() {
    std::vector<common::GpuInfo> gpus;
    if (!BuildLocalCudaGpuInfosV2(50000, &gpus)) {
        std::printf("BuildLocalCudaGpuInfosV2 failed\n");
        return 1;
    }

    for (const auto& gpu : gpus) {
        std::printf(
            "gpu=%d name=%s vendor=%s total=%llu free=%llu port=%u status=%u flags=0x%x arch=%s\n",
            gpu.local_gpu_id,
            gpu.gpu_name.c_str(),
            gpu_vendor_name(gpu.gpu_vendor),
            static_cast<unsigned long long>(gpu.total_mem_bytes),
            static_cast<unsigned long long>(gpu.free_mem_bytes),
            gpu.worker_port,
            static_cast<unsigned>(gpu.gpu_status),
            gpu.capability_flags,
            gpu.arch_name.c_str());
    }

    return 0;
}
