#pragma once

#include <cstdint>
#include <vector>

#include "common/types.h"

bool BuildLocalCpuFp16ResidentGpuInfosV2(
    std::int32_t worker_id_begin,
    std::uint32_t worker_port_base,
    std::vector<common::StaticGpuInfo>* out);

bool BuildLocalCpuFp16ResidentDynamicGpuInfosV2(
    std::int32_t worker_id_begin,
    std::vector<common::DynamicGpuInfo>* out);
