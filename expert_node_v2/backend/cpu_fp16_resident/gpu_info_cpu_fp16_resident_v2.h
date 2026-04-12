#pragma once

#include <vector>

#include "common/types.h"

namespace expert_node_v2 {

std::vector<common::StaticGpuInfo> BuildLocalCpuFp16ResidentGpuInfosV2();

std::vector<common::DynamicGpuInfo> BuildLocalCpuFp16ResidentDynamicGpuInfosV2();

}  // namespace expert_node_v2
