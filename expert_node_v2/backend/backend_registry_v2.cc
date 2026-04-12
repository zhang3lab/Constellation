#include "expert_node_v2/backend/backend_registry_v2.h"

#include "expert_node_v2/build_config_v2.h"

#if EXPERT_NODE_V2_ENABLE_CPU
#include "expert_node_v2/backend/cpu/backend_cpu_v2.h"
#include "expert_node_v2/backend/cpu/gpu_info_cpu_v2.h"
#endif

#if EXPERT_NODE_V2_ENABLE_CPU_FP16_RESIDENT
#include "expert_node_v2/backend/cpu_fp16_resident/backend_cpu_fp16_resident_v2.h"
#include "expert_node_v2/backend/cpu_fp16_resident/gpu_info_cpu_fp16_resident_v2.h"
#endif

#if EXPERT_NODE_V2_ENABLE_CUDA
#include "expert_node_v2/backend/cuda/backend_cuda_v2.h"
#include "expert_node_v2/backend/cuda/gpu_info_cuda_v2.h"
#endif

#if EXPERT_NODE_V2_ENABLE_AMD
#include "expert_node_v2/backend/amd/backend_amd_v2.h"
#include "expert_node_v2/backend/amd/gpu_info_amd_v2.h"
#endif

#if EXPERT_NODE_V2_ENABLE_INTEL
#include "expert_node_v2/backend/intel/backend_intel_v2.h"
#include "expert_node_v2/backend/intel/gpu_info_intel_v2.h"
#endif

namespace expert_node_v2 {

const std::array<BackendRegistryEntryV2, common::kGpuVendorCount>&
GetBackendRegistryV2() {
    static const std::array<BackendRegistryEntryV2, common::kGpuVendorCount>
        kRegistry = [] {
            std::array<BackendRegistryEntryV2, common::kGpuVendorCount> reg{};

            for (std::size_t i = 0; i < reg.size(); ++i) {
                reg[i].vendor = static_cast<common::GpuVendor>(i);
            }

#if EXPERT_NODE_V2_ENABLE_CPU
            reg[static_cast<std::size_t>(common::GpuVendor::Cpu)] = {
                .vendor = common::GpuVendor::Cpu,
                .build_static = BuildLocalCpuGpuInfosV2,
                .build_dynamic = BuildLocalCpuDynamicGpuInfosV2,
                .upload_expert = UploadExpertCpuV2,
                .free_expert_weights = FreeExpertWeightsCpuV2,
            };
#endif

#if EXPERT_NODE_V2_ENABLE_CPU_FP16_RESIDENT
            reg[static_cast<std::size_t>(common::GpuVendor::CpuFp16Resident)] = {
                .vendor = common::GpuVendor::CpuFp16Resident,
                .build_static = BuildLocalCpuFp16ResidentGpuInfosV2,
                .build_dynamic = BuildLocalCpuFp16ResidentDynamicGpuInfosV2,
                .upload_expert = UploadExpertCpuFp16ResidentV2,
                .free_expert_weights = FreeExpertWeightsCpuFp16ResidentV2,
            };
#endif

#if EXPERT_NODE_V2_ENABLE_CUDA
            reg[static_cast<std::size_t>(common::GpuVendor::Nvidia)] = {
                .vendor = common::GpuVendor::Nvidia,
                .build_static = BuildLocalCudaGpuInfosV2,
                .build_dynamic = BuildLocalCudaDynamicGpuInfosV2,
                .upload_expert = UploadExpertCudaV2,
                .free_expert_weights = FreeExpertWeightsCudaV2,
            };
#endif

#if EXPERT_NODE_V2_ENABLE_AMD
            reg[static_cast<std::size_t>(common::GpuVendor::AMD)] = {
                .vendor = common::GpuVendor::AMD,
                .build_static = BuildLocalAmdGpuInfosV2,
                .build_dynamic = BuildLocalAmdDynamicGpuInfosV2,
                .upload_expert = UploadExpertAmdV2,
                .free_expert_weights = FreeExpertWeightsAmdV2,
            };
#endif

#if EXPERT_NODE_V2_ENABLE_INTEL
            reg[static_cast<std::size_t>(common::GpuVendor::Intel)] = {
                .vendor = common::GpuVendor::Intel,
                .build_static = BuildLocalIntelGpuInfosV2,
                .build_dynamic = BuildLocalIntelDynamicGpuInfosV2,
                .upload_expert = UploadExpertIntelV2,
                .free_expert_weights = FreeExpertWeightsIntelV2,
            };
#endif

            return reg;
        }();

    return kRegistry;
}

const BackendRegistryEntryV2* FindBackendRegistryEntryV2(
    common::GpuVendor vendor) {
    const std::size_t idx = static_cast<std::size_t>(vendor);
    const auto& registry = GetBackendRegistryV2();
    if (idx >= registry.size()) {
        return nullptr;
    }

    const BackendRegistryEntryV2& entry = registry[idx];
    if (entry.vendor != vendor) {
        return nullptr;
    }
    if (entry.build_static == nullptr ||
        entry.build_dynamic == nullptr ||
        entry.upload_expert == nullptr ||
        entry.free_expert_weights == nullptr) {
        return nullptr;
    }
    return &entry;
}

}  // namespace expert_node_v2
