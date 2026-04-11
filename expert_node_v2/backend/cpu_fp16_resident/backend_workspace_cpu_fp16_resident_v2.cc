#include "expert_node_v2/backend/cpu_fp16_resident/backend_workspace_cpu_fp16_resident_v2.h"

#include <cstddef>
#include <memory>

#include "expert_node_v2/backend/cpu_fp16_resident/backend_cpu_fp16_resident_v2.h"

namespace expert_node_v2 {

class BackendWorkspaceCpuFp16ResidentV2::Impl {
public:
    int local_gpu_id = -1;
    ExpertWorkspaceCpuV2 ws{};
    bool ok = false;
};

namespace {

bool run_expert_request_cpu_fp16_resident(
    ExpertWorkspaceCpuV2* ws,
    const ExpertDeviceStorageV2& storage,
    const common::InferRequestMsg& req,
    common::InferResponseMsg* resp) {
    if (ws == nullptr || resp == nullptr) return false;

    if (req.batch_size <= 0 || req.hidden_dim <= 0) {
        return false;
    }

    const std::size_t elems =
        static_cast<std::size_t>(req.batch_size) *
        static_cast<std::size_t>(req.hidden_dim);

    std::size_t in_elem_bytes = 0;
    switch (req.input_dtype) {
        case common::ActivationDType::FP16:
        case common::ActivationDType::BF16:
            in_elem_bytes = 2;
            break;
        default:
            return false;
    }

    std::size_t out_elem_bytes = 0;
    switch (req.output_dtype) {
        case common::ActivationDType::FP16:
        case common::ActivationDType::BF16:
            out_elem_bytes = 2;
            break;
        default:
            return false;
    }

    const std::size_t in_bytes = elems * in_elem_bytes;
    const std::size_t out_bytes = elems * out_elem_bytes;

    if (req.activation.size() != in_bytes) {
        return false;
    }

    resp->status_code = 0;
    resp->batch_size = req.batch_size;
    resp->hidden_dim = req.hidden_dim;
    resp->output_dtype = req.output_dtype;
    resp->output.resize(out_bytes);

    return RunExpertCpuFp16ResidentV2(
        storage.view(),
        ws,
        req.activation.data(),
        req.input_dtype,
        resp->output.data(),
        req.output_dtype);
}

}  // namespace

BackendWorkspaceCpuFp16ResidentV2::BackendWorkspaceCpuFp16ResidentV2(
    int local_gpu_id,
    const ExpertWorkspaceConfigV2& config)
    : BackendWorkspaceV2(config),
      impl_(std::make_unique<Impl>()) {
    impl_->local_gpu_id = local_gpu_id;
    if (impl_->local_gpu_id < 0) {
        return;
    }

    impl_->ok = InitExpertWorkspaceCpuFp16ResidentV2(config_, &impl_->ws);
}

BackendWorkspaceCpuFp16ResidentV2::~BackendWorkspaceCpuFp16ResidentV2() {
    if (impl_ && impl_->ok) {
        FreeExpertWorkspaceCpuFp16ResidentV2(&impl_->ws);
    }
}

bool BackendWorkspaceCpuFp16ResidentV2::RunExpertRequest(
    const ExpertDeviceStorageV2& storage,
    const common::InferRequestMsg& req,
    common::InferResponseMsg* resp) {
    if (!impl_ || !impl_->ok || resp == nullptr) return false;
    if (impl_->local_gpu_id < 0) return false;

    if (req.batch_size != 1) return false;
    if (req.hidden_dim != config_.hidden_dim) return false;

    return run_expert_request_cpu_fp16_resident(&impl_->ws, storage, req, resp);
}

bool BackendWorkspaceCpuFp16ResidentV2::ok() const {
    return impl_ && impl_->ok;
}

}  // namespace expert_node_v2
