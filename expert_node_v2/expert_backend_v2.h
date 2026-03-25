#pragma once

#include "expert_node_v2/expert_format_v2.h"
#include "expert_node_v2/cuda/backend_cuda_v2.h"

namespace expert_node_v2 {

using ExpertWorkspaceV2 = ExpertWorkspaceCudaV2;

bool UploadExpertForGpuV2(
    int local_gpu_id,
    const ExpertTensorBundleV2& host_bundle,
    ExpertDeviceStorageV2* out_storage);

void FreeExpertWeightsV2(ExpertDeviceStorageV2* storage);

bool InitExpertWorkspaceV2(
    const ExpertWorkspaceConfigV2& config,
    ExpertWorkspaceV2* out_ws);

void FreeExpertWorkspaceV2(ExpertWorkspaceV2* ws);

template <class TAct>
inline bool RunExpertV2(
    const ExpertWeightsViewV2& expert_device_view,
    ExpertWorkspaceV2* ws,
    const TAct* d_x,
    TAct* d_y,
    cudaStream_t stream) {
    return RunExpertCudaV2<TAct>(expert_device_view, ws, d_x, d_y, stream);
}

}  // namespace expert_node_v2
