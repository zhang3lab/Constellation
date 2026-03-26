#pragma once

#include <memory>

#include "common/messages.h"
#include "common/node_info.h"
#include "expert_node_v2/expert_format_v2.h"

namespace expert_node_v2 {

class BackendWorkspaceV2 {
public:
    virtual ~BackendWorkspaceV2() = default;

    virtual bool RunExpertRequest(
        const ExpertDeviceStorageV2& storage,
        const common::InferRequestMsg& req,
        common::InferResponseMsg* resp) = 0;
};

std::unique_ptr<BackendWorkspaceV2> CreateBackendWorkspaceV2(
    const common::VendorWorkerSpan& vendor_span,
    int worker_id);

}  // namespace expert_node_v2
