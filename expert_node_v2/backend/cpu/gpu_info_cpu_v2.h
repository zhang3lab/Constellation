#pragma once

#include <vector>

#include "common/types.h"

namespace expert_node_v2 {

bool QueryGpuInfoCpuV2(
    int num_cpu_workers,
    std::vector<common::StaticGpuInfo>* out_gpus);

}  // namespace expert_node_v2
