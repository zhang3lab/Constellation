#pragma once

#include "expert_node_v2/expert_format_v2.h"

namespace expert_node_v2 {

inline bool UploadExpertIntelV2(
    int local_gpu_id,
    const ExpertTensorBundleV2& host_bundle,
    ExpertDeviceStorageV2* out_storage) {
    (void)local_gpu_id;
    (void)host_bundle;

    if (out_storage != nullptr) {
        *out_storage = ExpertDeviceStorageV2{};
    }
    return false;
}

inline void FreeExpertWeightsIntelV2(ExpertDeviceStorageV2* storage) {
    if (storage == nullptr) return;
    *storage = ExpertDeviceStorageV2{};

}

}  // namespace expert_node_v2
