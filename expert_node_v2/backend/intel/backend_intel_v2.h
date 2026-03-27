#pragma once

#include "expert_node_v2/expert_format_v2.h"

namespace expert_node_v2 {

inline bool UploadExpertIntelV2(
    int local_gpu_id,
    const ExpertTensorBundleV2& host_bundle,
    ExpertDeviceStorageV2* out_storage) {
    (void)local_gpu_id;
    (void)host_bundle;
    (void)out_storage;
    return false;
}

inline void FreeExpertWeightsIntelV2(ExpertDeviceStorageV2* storage) {
    if (storage != nullptr) {
        storage->clear();
    }
}

}  // namespace expert_node_v2
