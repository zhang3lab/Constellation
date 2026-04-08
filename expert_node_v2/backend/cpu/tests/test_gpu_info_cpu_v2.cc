#include <cstdio>
#include <vector>

#include "expert_node_v2/backend/cpu/gpu_info_cpu_v2.h"

int main() {
    constexpr int kWorkerIdBegin = 0;
    constexpr int kWorkerPortBase = 50000;
    constexpr int kNumCpuWorkers = 1;

    std::vector<common::StaticGpuInfo> static_gpus;
    if (!BuildLocalCpuGpuInfosV2(
            kWorkerIdBegin,
            kWorkerPortBase,
            &static_gpus)) {
        std::printf("BuildLocalCpuGpuInfosV2 failed\n");
        return 1;
    }

    std::vector<common::DynamicGpuInfo> dynamic_gpus;
    if (!BuildLocalCpuDynamicGpuInfosV2(
            kWorkerIdBegin,
            &dynamic_gpus)) {
        std::printf("BuildLocalCpuDynamicGpuInfosV2 failed\n");
        return 1;
    }

    std::printf("static_gpus=%zu dynamic_gpus=%zu\n",
                static_gpus.size(), dynamic_gpus.size());

    if (static_gpus.size() != static_cast<std::size_t>(kNumCpuWorkers)) {
        std::printf("unexpected static gpu count got=%zu expect=%d\n",
                    static_gpus.size(), kNumCpuWorkers);
        return 1;
    }

    if (dynamic_gpus.size() != static_cast<std::size_t>(kNumCpuWorkers)) {
        std::printf("unexpected dynamic gpu count got=%zu expect=%d\n",
                    dynamic_gpus.size(), kNumCpuWorkers);
        return 1;
    }

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

        if (gpu.gpu_vendor != common::GpuVendor::Cpu) {
            std::printf("unexpected vendor for worker_id=%d\n", gpu.worker_id);
            return 1;
        }
        if (gpu.worker_id < 0) {
            std::printf("unexpected worker_id=%d\n", gpu.worker_id);
            return 1;
        }
        if (gpu.gpu_name.empty()) {
            std::printf("empty cpu gpu_name for worker_id=%d\n", gpu.worker_id);
            return 1;
        }
        if (gpu.total_mem_bytes == 0) {
            std::printf("total_mem_bytes is zero for worker_id=%d\n", gpu.worker_id);
            return 1;
        }
    }

    for (const auto& gpu : dynamic_gpus) {
        std::printf(
            "[dynamic] worker_id=%d free=%llu status=%s\n",
            gpu.worker_id,
            static_cast<unsigned long long>(gpu.free_mem_bytes),
            common::gpu_status_name(gpu.gpu_status));

        if (gpu.worker_id < 0) {
            std::printf("unexpected dynamic worker_id=%d\n", gpu.worker_id);
            return 1;
        }
        if (gpu.free_mem_bytes == 0) {
            std::printf("free_mem_bytes is zero for worker_id=%d\n", gpu.worker_id);
            return 1;
        }
        if (gpu.gpu_status != common::GpuStatus::Idle) {
            std::printf("unexpected gpu_status for worker_id=%d\n", gpu.worker_id);
            return 1;
        }
    }

    for (std::size_t i = 0; i < static_gpus.size(); ++i) {
        if (static_gpus[i].worker_id != dynamic_gpus[i].worker_id) {
            std::printf("worker_id mismatch at i=%zu static=%d dynamic=%d\n",
                        i,
                        static_gpus[i].worker_id,
                        dynamic_gpus[i].worker_id);
            return 1;
        }
        if (dynamic_gpus[i].free_mem_bytes > static_gpus[i].total_mem_bytes) {
            std::printf(
                "free_mem_bytes > total_mem_bytes at i=%zu free=%llu total=%llu\n",
                i,
                static_cast<unsigned long long>(dynamic_gpus[i].free_mem_bytes),
                static_cast<unsigned long long>(static_gpus[i].total_mem_bytes));
            return 1;
        }
    }

    return 0;
}
