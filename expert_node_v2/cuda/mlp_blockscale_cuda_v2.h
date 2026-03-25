#pragma once

#include "expert_node_v2/build_config_v2.h"

#include <cuda_fp16.h>
#if EXPERT_NODE_V2_HAS_CUDA_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_runtime.h>

#include "expert_node_v2/expert_format_v2.h"

template <class TAct>
bool LaunchFusedUpGateCudaV2Impl(
    const MatrixBlockScaleViewV2& w_up_device_view,
    const MatrixBlockScaleViewV2& w_gate_device_view,
    const TAct* d_x,
    float* d_h,
    cudaStream_t stream);

template <class TAct>
bool LaunchDownCudaV2Impl(
    const MatrixBlockScaleViewV2& w_down_device_view,
    const float* d_h,
    TAct* d_y,
    cudaStream_t stream);
