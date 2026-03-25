#pragma once

#include "expert_node_v2/build_config_v2.h"

#include <cuda_fp16.h>
#if EXPERT_NODE_V2_HAS_CUDA_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_runtime.h>

#include "expert_node_v2/expert_format_v2.h"

template <class TIn, class TOut>
bool LaunchMatvecBlockScaleCudaV2(
    const MatrixBlockScaleViewV2& W,
    const TIn* d_x,
    TOut* d_y,
    cudaStream_t stream);
