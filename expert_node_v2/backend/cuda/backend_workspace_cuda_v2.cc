#include "expert_node_v2/backend/cuda/backend_workspace_cuda_v2.h"

#include <memory>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if EXPERT_NODE_V2_HAS_CUDA_BF16
#include <cuda_bf16.h>
#endif

#include "expert_node_v2/backend/cuda/backend_cuda_v2.h"

namespace expert_node_v2 {

class BackendWorkspaceCudaV2::Impl {
public:
    int local_gpu_id = -1;
    ExpertWorkspaceCudaV2 ws{};
    bool ok = false;
};

namespace {

template <class TIn, class TOut>
bool run_expert_request_typed(
    ExpertWorkspaceCudaV2* ws,
    const ExpertDeviceStorageV2& storage,
    const common::InferRequestMsg& req,
    common::InferResponseMsg* resp) {
    if (ws == nullptr || resp == nullptr) return false;

    const size_t elems =
        static_cast<size_t>(req.batch_size) * static_cast<size_t>(req.hidden_dim);
    const size_t in_bytes = elems * sizeof(TIn);
    const size_t out_bytes = elems * sizeof(TOut);

    if (req.activation.size() != in_bytes) {
        return false;
    }

    TIn* d_input = nullptr;
    TOut* d_output = nullptr;

    auto cleanup = [&]() {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        d_input = nullptr;
        d_output = nullptr;
    };

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_input), in_bytes);
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_output), out_bytes);
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }

    err = cudaMemcpy(
        d_input,
        req.activation.data(),
        in_bytes,
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }

    const bool ok = RunExpertCudaV2<TIn, TOut>(
        storage.view(),
        ws,
        d_input,
        d_output,
        0);
    if (!ok) {
        cleanup();
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }

    resp->status_code = 0;
    resp->batch_size = req.batch_size;
    resp->hidden_dim = req.hidden_dim;
    resp->output_dtype = req.output_dtype;
    resp->output.resize(out_bytes);

    err = cudaMemcpy(
        resp->output.data(),
        d_output,
        out_bytes,
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }

    cleanup();
    return true;
}

}  // namespace

BackendWorkspaceCudaV2::BackendWorkspaceCudaV2(
    int local_gpu_id,
    const ExpertWorkspaceConfigV2& config)
    : BackendWorkspaceV2(config),
      impl_(std::make_unique<Impl>()) {
    impl_->local_gpu_id = local_gpu_id;
    if (impl_->local_gpu_id < 0) {
        return;
    }

    cudaError_t err = cudaSetDevice(impl_->local_gpu_id);
    if (err != cudaSuccess) {
        impl_->local_gpu_id = -1;
        return;
    }

    impl_->ok = InitExpertWorkspaceCudaV2(config_, &impl_->ws);
}

BackendWorkspaceCudaV2::~BackendWorkspaceCudaV2() {
    if (impl_ && impl_->ok && impl_->local_gpu_id >= 0) {
        cudaSetDevice(impl_->local_gpu_id);
        FreeExpertWorkspaceCudaV2(&impl_->ws);
    }
}

bool BackendWorkspaceCudaV2::RunExpertRequest(
    const ExpertDeviceStorageV2& storage,
    const common::InferRequestMsg& req,
    common::InferResponseMsg* resp) {
    if (!impl_ || !impl_->ok || resp == nullptr) return false;
    if (impl_->local_gpu_id < 0) return false;

    if (req.batch_size != 1) return false;
    if (req.hidden_dim != config_.hidden_dim) return false;

    const cudaError_t err = cudaSetDevice(impl_->local_gpu_id);
    if (err != cudaSuccess) {
        return false;
    }

    if (req.input_dtype == common::ActivationDType::FP16 &&
        req.output_dtype == common::ActivationDType::FP16) {
        return run_expert_request_typed<__half, __half>(
            &impl_->ws, storage, req, resp);
    }

#if EXPERT_NODE_V2_HAS_CUDA_BF16
    if (req.input_dtype == common::ActivationDType::FP16 &&
        req.output_dtype == common::ActivationDType::BF16) {
        return run_expert_request_typed<__half, __nv_bfloat16>(
            &impl_->ws, storage, req, resp);
    }

    if (req.input_dtype == common::ActivationDType::BF16 &&
        req.output_dtype == common::ActivationDType::FP16) {
        return run_expert_request_typed<__nv_bfloat16, __half>(
            &impl_->ws, storage, req, resp);
    }

    if (req.input_dtype == common::ActivationDType::BF16 &&
        req.output_dtype == common::ActivationDType::BF16) {
        return run_expert_request_typed<__nv_bfloat16, __nv_bfloat16>(
            &impl_->ws, storage, req, resp);
    }
#endif

    return false;
}

bool BackendWorkspaceCudaV2::ok() const {
    return impl_ && impl_->ok;
}

}  // namespace expert_node_v2
