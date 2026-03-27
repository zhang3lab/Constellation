#pragma once

#include <cuda_runtime.h>

#include "expert_node_v2/expert_format_v2.h"

const float* GetOrInitFp8DecodeLutCudaV2(
    int device_id,
    Fp8Format fmt,
    cudaStream_t stream);

void ResetFp8DecodeLutsCudaV2();
