#pragma once

#include "common/protocol.h"
#include "expert_node_v2/expert_format_v2.h"

bool RunDownCpuV2(
    const MatrixBlockScaleViewV2& w_down,
    const float* h,
    void* y,
    common::ActivationDType output_dtype);
