#pragma once

#include "expert_node_v2/build_config_v2.h"

#include <cuda_fp16.h>
#if EXPERT_NODE_V2_HAS_CUDA_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_runtime.h>

#include "expert_node_v2/expert_format_v2.h"

struct ExpertWorkspaceCudaV2 {
    DeviceBufferV2<float> d_tmp;

    void clear() {
        d_tmp.clear();
    }
};

bool UploadExpertCudaV2(
    int local_gpu_id,
    const ExpertTensorBundleV2& host_bundle,
    ExpertDeviceStorageV2* out_storage);

void FreeExpertWeightsCudaV2(ExpertDeviceStorageV2* storage);

bool InitExpertWorkspaceCudaV2(
    const ExpertWorkspaceConfigV2& config,
    ExpertWorkspaceCudaV2* out_ws);

void FreeExpertWorkspaceCudaV2(ExpertWorkspaceCudaV2* ws);

template <class TIn, class TOut>
bool RunExpertCudaV2(
    const ExpertWeightsViewV2& expert_device_view,
    ExpertWorkspaceCudaV2* ws,
    const TIn* d_x,
    TOut* d_y,
    cudaStream_t stream);
