#pragma once

#include <memory>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if EXPERT_NODE_V2_HAS_CUDA_BF16
#include <cuda_bf16.h>
#endif

#include "expert_node_v2/cuda/backend_cuda_v2.h"
#include "expert_node_v2/cuda/backend_workspace_cuda_v2.h"

namespace expert_node_v2 {

class BackendWorkspaceCudaV2::Impl {
public:
    int local_gpu_id = -1;
    ExpertWorkspaceCudaV2 ws{};
    bool ok = false;
};

}  // namespace expert_node_v2
