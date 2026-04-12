#include "expert_node_v2/backend/cpu_fp16_resident/gpu_info_cpu_fp16_resident_v2.h"

#include "expert_node_v2/backend/cpu/gpu_info_cpu_v2.h"

bool BuildLocalCpuFp16ResidentGpuInfosV2(
    std::int32_t worker_id_begin,
    std::uint32_t worker_port_base,
    std::vector<common::StaticGpuInfo>* out) {
    if (out == nullptr) return false;

    if (!BuildLocalCpuGpuInfosV2(worker_id_begin, worker_port_base, out)) {
        return false;
    }

    for (auto& info : *out) {
        info.gpu_vendor = common::GpuVendor::CpuFp16Resident;
    }

    return true;
}

bool BuildLocalCpuFp16ResidentDynamicGpuInfosV2(
    std::int32_t worker_id_begin,
    std::vector<common::DynamicGpuInfo>* out) {
    return BuildLocalCpuDynamicGpuInfosV2(worker_id_begin, out);
}
