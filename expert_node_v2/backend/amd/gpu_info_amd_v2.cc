#include "expert_node_v2/backend/amd/gpu_info_amd_v2.h"

bool BuildLocalAmdGpuInfosV2(
    std::int32_t worker_id_begin,
    std::uint32_t worker_port_base,
    std::vector<common::StaticGpuInfo>* out) {
    (void)worker_id_begin;
    (void)worker_port_base;
    if (out == nullptr) return false;
    out->clear();
    return true;
}

bool BuildLocalAmdDynamicGpuInfosV2(
    std::int32_t worker_id_begin,
    std::vector<common::DynamicGpuInfo>* out) {
    (void)worker_id_begin;
    if (out == nullptr) return false;
    out->clear();
    return true;
}
