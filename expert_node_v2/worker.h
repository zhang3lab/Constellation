#pragma once

#include <memory>
#include <string>

#include "common/protocol.h"
#include "expert_node_v2/backend/backend_workspace_v2.h"

struct ControlState;

struct GpuWorkerContextV2 {
    int worker_id = -1;
    int worker_port = 0;
    ControlState* state = nullptr;

    std::unique_ptr<expert_node_v2::BackendWorkspaceV2> workspace;
};

bool HandleInferRequest(
    int fd,
    GpuWorkerContextV2* ctx,
    const common::MsgHeader& req,
    const std::string& req_body);

void RunGpuWorkerLoopV2(GpuWorkerContextV2* ctx);
