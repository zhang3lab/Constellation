#include "expert_node_v2/backend/cpu_fp16_resident/gpu_info_cpu_fp16_resident_v2.h"

#include "expert_node_v2/backend/cpu/gpu_info_cpu_v2.h"

namespace expert_node_v2 {

std::vector<common::StaticGpuInfo> BuildLocalCpuFp16ResidentGpuInfosV2() {
    std::vector<common::StaticGpuInfo> infos = BuildLocalCpuGpuInfosV2();
    for (auto& info : infos) {
        info.gpu_vendor = common::GpuVendor::CpuFp16Resident;
    }
    return infos;
}

std::vector<common::DynamicGpuInfo> BuildLocalCpuFp16ResidentDynamicGpuInfosV2() {
    return BuildLocalCpuDynamicGpuInfosV2();
}

}  // namespace expert_node_v2
