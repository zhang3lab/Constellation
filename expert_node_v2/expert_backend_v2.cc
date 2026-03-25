#include "expert_node_v2/expert_backend_v2.h"

#include <cuda_runtime.h>

namespace expert_node_v2 {

bool UploadExpertForGpuV2(
    int local_gpu_id,
    const ExpertTensorBundleV2& host_bundle,
    ExpertDeviceStorageV2* out_storage) {
    cudaError_t err = cudaSetDevice(local_gpu_id);
    if (err != cudaSuccess) {
        return false;
    }
    return UploadExpertCudaV2(host_bundle, out_storage);
}

void FreeExpertWeightsV2(ExpertDeviceStorageV2* storage) {
    FreeExpertWeightsCudaV2(storage);
}

bool InitExpertWorkspaceV2(
    const ExpertWorkspaceConfigV2& config,
    ExpertWorkspaceV2* out_ws) {
    return InitExpertWorkspaceCudaV2(config, out_ws);
}

void FreeExpertWorkspaceV2(ExpertWorkspaceV2* ws) {
    FreeExpertWorkspaceCudaV2(ws);
}

}  // namespace expert_node_v2
