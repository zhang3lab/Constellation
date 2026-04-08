#include <cstdio>
#include <vector>

#include "expert_node_v2/backend/cpu/gpu_info_cpu_v2.h"

int main() {
    constexpr int kNumCpuWorkers = 1;

    std::vector<common::StaticGpuInfo> gpus;
    if (!expert_node_v2::QueryGpuInfoCpuV2(kNumCpuWorkers, &gpus)) {
        std::printf("QueryGpuInfoCpuV2 failed\n");
        return 1;
    }

    std::printf("cpu_gpus=%zu\n", gpus.size());

    if (gpus.size() != static_cast<std::size_t>(kNumCpuWorkers)) {
        std::printf(
            "unexpected gpu count got=%zu expect=%d\n",
            gpus.size(),
            kNumCpuWorkers);
        return 1;
    }

    for (const auto& gpu : gpus) {
        std::printf(
            "[cpu] worker_id=%d name=%s vendor=%s total=%llu free=%llu\n",
            gpu.worker_id,
            gpu.name.c_str(),
            common::gpu_vendor_name(gpu.gpu_vendor),
            static_cast<unsigned long long>(gpu.total_bytes),
            static_cast<unsigned long long>(gpu.free_bytes));

        if (gpu.gpu_vendor != common::GpuVendor::Cpu) {
            std::printf("unexpected vendor for worker_id=%d\n", gpu.worker_id);
            return 1;
        }

        if (gpu.worker_id < 0) {
            std::printf("unexpected worker_id=%d\n", gpu.worker_id);
            return 1;
        }

        if (gpu.name.empty()) {
            std::printf("empty cpu name for worker_id=%d\n", gpu.worker_id);
            return 1;
        }

        if (gpu.total_bytes == 0) {
            std::printf("total_bytes is zero for worker_id=%d\n", gpu.worker_id);
            return 1;
        }

        if (gpu.free_bytes == 0) {
            std::printf("free_bytes is zero for worker_id=%d\n", gpu.worker_id);
            return 1;
        }

        if (gpu.free_bytes > gpu.total_bytes) {
            std::printf(
                "free_bytes > total_bytes for worker_id=%d free=%llu total=%llu\n",
                gpu.worker_id,
                static_cast<unsigned long long>(gpu.free_bytes),
                static_cast<unsigned long long>(gpu.total_bytes));
            return 1;
        }
    }

    return 0;
}
