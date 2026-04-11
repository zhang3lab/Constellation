#pragma once

#include <memory>

#include "expert_node_v2/backend/backend_workspace_v2.h"

namespace expert_node_v2 {

class BackendWorkspaceCpuFp16ResidentV2 final : public BackendWorkspaceV2 {
public:
    BackendWorkspaceCpuFp16ResidentV2(
        int local_gpu_id,
        const ExpertWorkspaceConfigV2& config);
    ~BackendWorkspaceCpuFp16ResidentV2() override;

    bool RunExpertRequest(
        const ExpertDeviceStorageV2& storage,
        const common::InferRequestMsg& req,
        common::InferResponseMsg* resp) override;

    bool ok() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace expert_node_v2
