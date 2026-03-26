#pragma once

#include "expert_node_v2/backend_workspace_v2.h"
#include "expert_node_v2/cuda/backend_cuda_v2.h"

namespace expert_node_v2 {

class BackendWorkspaceCudaV2 final : public BackendWorkspaceV2 {
public:
    explicit BackendWorkspaceCudaV2(int local_gpu_id);
    ~BackendWorkspaceCudaV2() override;

    bool RunExpertRequest(
        const ExpertDeviceStorageV2& storage,
        const common::InferRequestMsg& req,
        common::InferResponseMsg* resp) override;

    bool ok() const { return ok_; }

private:
    int local_gpu_id_ = -1;
    ExpertWorkspaceCudaV2 ws_{};
    bool ok_ = false;
};

}  // namespace expert_node_v2
