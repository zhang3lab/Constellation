#include "expert_node_v2/backend_workspace_v2.h"

#include "expert_node_v2/build_config_v2.h"

#if EXPERT_NODE_V2_ENABLE_CUDA
#include "expert_node_v2/cuda/backend_workspace_cuda_v2.h"
#endif

namespace expert_node_v2 {

std::unique_ptr<BackendWorkspaceV2> CreateBackendWorkspaceV2(
    const common::VendorWorkerSpan& vendor_span,
    int worker_id) {
    if (worker_id < vendor_span.worker_id_begin) {
        return nullptr;
    }
    if (worker_id >= vendor_span.worker_id_begin + vendor_span.worker_count) {
        return nullptr;
    }

    const int local_gpu_id = worker_id - vendor_span.worker_id_begin;

    switch (vendor_span.vendor) {
#if EXPERT_NODE_V2_ENABLE_CUDA
        case common::GpuVendor::Nvidia: {
            auto p = std::make_unique<BackendWorkspaceCudaV2>(local_gpu_id);
            if (!p->ok()) {
                return nullptr;
            }
            return p;
        }
#endif
        default:
            return nullptr;
    }
}

}  // namespace expert_node_v2
