#pragma once

#include <cstdint>
#include <vector>

#include "common/types.h"

bool BuildLocalIntelGpuInfosV2(
    std::int32_t worker_id_begin,
    std::uint32_t worker_port_base,
    std::vector<common::StaticGpuInfo>* out);

bool BuildLocalIntelDynamicGpuInfosV2(
    std::int32_t worker_id_begin,
    std::vector<common::DynamicGpuInfo>* out);
