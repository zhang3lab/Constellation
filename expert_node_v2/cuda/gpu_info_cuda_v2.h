#pragma once

#include <cstdint>
#include <vector>

#include "common/types.h"

bool BuildLocalCudaGpuInfosV2(
    std::uint32_t base_worker_port,
    std::vector<common::GpuInfo>* out);
