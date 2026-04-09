#pragma once

#include "common/protocol.h"
#include "expert_node_v2/build_config_v2.h"
#include "expert_node_v2/expert_format_v2.h"

struct ExpertWorkspaceCpuV2 {
    DeviceBufferV2<float> tmp;
};

bool UploadExpertCpuV2(
    int local_gpu_id,
    const ExpertTensorBundleV2& host_bundle,
    ExpertDeviceStorageV2* out_storage);

void FreeExpertWeightsCpuV2(ExpertDeviceStorageV2* storage);

bool InitExpertWorkspaceCpuV2(
    const ExpertWorkspaceConfigV2& config,
    ExpertWorkspaceCpuV2* out_ws);

void FreeExpertWorkspaceCpuV2(ExpertWorkspaceCpuV2* ws);

bool RunExpertCpuV2(
    const ExpertWeightsViewV2& expert_device_view,
    ExpertWorkspaceCpuV2* ws,
    const void* x,
    common::ActivationDType input_dtype,
    void* y,
    common::ActivationDType output_dtype);
