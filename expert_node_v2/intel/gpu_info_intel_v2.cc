#include "expert_node_v2/intel/gpu_info_intel_v2.h"

bool BuildLocalIntelGpuInfosV2(
    std::int32_t worker_id_begin,
    std::uint32_t worker_port_base,
    std::vector<common::StaticGpuInfo>* out) {
    (void)worker_id_begin;
    (void)worker_port_base;
    if (out == nullptr) return false;
    out->clear();
    return true;
}

bool BuildLocalIntelDynamicGpuInfosV2(
    std::int32_t worker_id_begin,
    std::vector<common::DynamicGpuInfo>* out) {
    (void)worker_id_begin;
    if (out == nullptr) return false;
    out->clear();
    return true;
}
