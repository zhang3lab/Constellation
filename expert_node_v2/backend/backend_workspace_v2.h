#pragma once

#include <memory>

#include "common/infer_codec.h"
#include "common/types.h"
#include "expert_node_v2/expert_format_v2.h"

namespace expert_node_v2 {

class BackendWorkspaceV2 {
public:
    explicit BackendWorkspaceV2(const ExpertWorkspaceConfigV2& config)
        : config_(config) {}

    virtual ~BackendWorkspaceV2() = default;

    virtual bool RunExpertRequest(
        const ExpertDeviceStorageV2& storage,
        const common::InferRequestMsg& req,
        common::InferResponseMsg* resp) = 0;

    const ExpertWorkspaceConfigV2& config() const {
        return config_;
    }

protected:
    ExpertWorkspaceConfigV2 config_;
};

std::unique_ptr<BackendWorkspaceV2> CreateBackendWorkspaceV2(
    common::GpuVendor vendor,
    const common::VendorWorkerSpan& vendor_span,
    int worker_id);

}  // namespace expert_node_v2
