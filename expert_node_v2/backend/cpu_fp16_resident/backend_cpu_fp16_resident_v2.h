#pragma once

#include "common/protocol.h"
#include "expert_node_v2/backend/cpu/backend_cpu_v2.h"
#include "expert_node_v2/expert_format_v2.h"

bool UploadExpertCpuFp16ResidentV2(
    int local_gpu_id,
    const ExpertTensorBundleV2& host_bundle,
    ExpertDeviceStorageV2* out_storage);

void FreeExpertWeightsCpuFp16ResidentV2(
    ExpertDeviceStorageV2* storage);

bool InitExpertWorkspaceCpuFp16ResidentV2(
    const ExpertWorkspaceConfigV2& config,
    ExpertWorkspaceCpuV2* out_ws);

void FreeExpertWorkspaceCpuFp16ResidentV2(
    ExpertWorkspaceCpuV2* ws);

bool RunExpertCpuFp16ResidentV2(
    const ExpertWeightsViewV2& expert_device_view,
    ExpertWorkspaceCpuV2* ws,
    const void* x,
    common::ActivationDType input_dtype,
    void* y,
    common::ActivationDType output_dtype);
