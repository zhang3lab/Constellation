#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "common/types.h"
#include "expert_node_v2/expert_format_v2.h"

namespace expert_node_v2 {

using BuildStaticGpuInfosFnV2 = bool (*)(
    std::int32_t worker_id_begin,
    std::uint32_t worker_port_base,
    std::vector<common::StaticGpuInfo>* out);

using BuildDynamicGpuInfosFnV2 = bool (*)(
    std::int32_t worker_id_begin,
    std::vector<common::DynamicGpuInfo>* out);

using UploadExpertFnV2 = bool (*)(
    int local_gpu_id,
    const ExpertTensorBundleV2& host_bundle,
    ExpertDeviceStorageV2* out_storage);

using FreeExpertWeightsFnV2 = void (*)(
    ExpertDeviceStorageV2* storage);

struct BackendRegistryEntryV2 {
    common::GpuVendor vendor = common::GpuVendor::Unknown;
    BuildStaticGpuInfosFnV2 build_static = nullptr;
    BuildDynamicGpuInfosFnV2 build_dynamic = nullptr;
    UploadExpertFnV2 upload_expert = nullptr;
    FreeExpertWeightsFnV2 free_expert_weights = nullptr;
};

const std::array<BackendRegistryEntryV2, common::kGpuVendorCount>&
GetBackendRegistryV2();

const BackendRegistryEntryV2* FindBackendRegistryEntryV2(
    common::GpuVendor vendor);

}  // namespace expert_node_v2
