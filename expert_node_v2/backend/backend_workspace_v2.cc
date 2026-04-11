#include "expert_node_v2/backend/backend_workspace_v2.h"

#include "expert_node_v2/build_config_v2.h"

#if EXPERT_NODE_V2_ENABLE_CPU
#include "expert_node_v2/backend/cpu/backend_workspace_cpu_v2.h"
#endif

#if EXPERT_NODE_V2_ENABLE_CPU_FP16_RESIDENT
#include "expert_node_v2/backend/cpu_fp16_resident/backend_workspace_cpu_fp16_resident_v2.h"
#endif

#if EXPERT_NODE_V2_ENABLE_CUDA
#include "expert_node_v2/backend/cuda/backend_workspace_cuda_v2.h"
#endif

namespace expert_node_v2 {

namespace {

ExpertWorkspaceConfigV2 MakeDefaultWorkspaceConfigV2() {
    ExpertWorkspaceConfigV2 cfg;
    cfg.hidden_dim = 7168;
    cfg.inter_dim = 2048;
    return cfg;
}

}  // namespace

std::unique_ptr<BackendWorkspaceV2> CreateBackendWorkspaceV2(
    common::GpuVendor vendor,
    const common::VendorWorkerSpan& vendor_span,
    int worker_id) {
    if (worker_id < vendor_span.worker_id_begin) {
        return nullptr;
    }
    if (worker_id >= vendor_span.worker_id_begin + vendor_span.worker_count) {
        return nullptr;
    }

    const int local_gpu_id = worker_id - vendor_span.worker_id_begin;
    const ExpertWorkspaceConfigV2 cfg = MakeDefaultWorkspaceConfigV2();

    switch (vendor) {
#if EXPERT_NODE_V2_ENABLE_CPU
        case common::GpuVendor::Cpu: {
            auto p = std::make_unique<BackendWorkspaceCpuV2>(
                local_gpu_id, cfg);
            if (!p->ok()) {
                return nullptr;
            }
            return p;
        }
#endif

#if EXPERT_NODE_V2_ENABLE_CPU_FP16_RESIDENT
        case common::GpuVendor::CpuFp16Resident: {
            auto p = std::make_unique<BackendWorkspaceCpuFp16ResidentV2>(
                local_gpu_id, cfg);
            if (!p->ok()) {
                return nullptr;
            }
            return p;
        }
#endif

#if EXPERT_NODE_V2_ENABLE_CUDA
        case common::GpuVendor::Nvidia: {
            auto p = std::make_unique<BackendWorkspaceCudaV2>(
                local_gpu_id, cfg);
            if (!p->ok()) {
                return nullptr;
            }
            return p;
        }
#endif

#if EXPERT_NODE_V2_ENABLE_AMD
        case common::GpuVendor::AMD:
            return nullptr;
#endif

#if EXPERT_NODE_V2_ENABLE_INTEL
        case common::GpuVendor::Intel:
            return nullptr;
#endif

        default:
            return nullptr;
    }
}

}  // namespace expert_node_v2
