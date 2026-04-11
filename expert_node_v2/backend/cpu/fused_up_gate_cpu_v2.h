#pragma once

#include "common/protocol.h"
#include "expert_node_v2/expert_format_v2.h"

bool RunFusedUpGateCpuV2(
    const MatrixBlockScaleViewV2& w_up,
    const MatrixBlockScaleViewV2& w_gate,
    const void* x,
    common::ActivationDType input_dtype,
    float* h,
    int omp_threads);
