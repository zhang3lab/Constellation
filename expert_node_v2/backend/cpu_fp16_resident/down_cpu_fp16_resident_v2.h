#pragma once

#include "common/protocol.h"
#include "expert_node_v2/expert_format_v2.h"


// For cpu_fp16_resident backend, w_down.weight.data stores row-major FP16
// resident weights directly, not FP8 blockscale payload.
bool RunDownCpuFp16ResidentV2(
    const MatrixBlockScaleViewV2& w_down,
    const float* h,
    void* y,
    common::ActivationDType output_dtype,
    int omp_threads = 1);
