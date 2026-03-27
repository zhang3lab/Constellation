#pragma once

#include "expert_node_v2/build_config_v2.h"

#include <cuda_fp16.h>
#if EXPERT_NODE_V2_HAS_CUDA_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_runtime.h>

__device__ inline float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template <class T>
__device__ inline float act_to_float(T x);

template <>
__device__ inline float act_to_float<float>(float x) {
    return x;
}

template <>
__device__ inline float act_to_float<__half>(__half x) {
    return __half2float(x);
}

#if EXPERT_NODE_V2_HAS_CUDA_BF16
template <>
__device__ inline float act_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
#endif

template <class T>
__device__ inline T float_to_act(float x);

template <>
__device__ inline float float_to_act<float>(float x) {
    return x;
}

template <>
__device__ inline __half float_to_act<__half>(float x) {
    return __float2half(x);
}

#if EXPERT_NODE_V2_HAS_CUDA_BF16
template <>
__device__ inline __nv_bfloat16 float_to_act<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}
#endif
