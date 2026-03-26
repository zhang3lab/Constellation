#include "expert_node_v2/cuda/backend_workspace_cuda_v2.h"

#include <memory>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if EXPERT_NODE_V2_HAS_CUDA_BF16
#include <cuda_bf16.h>
#endif

namespace expert_node_v2 {

namespace {

template <class TIn, class TOut>
bool run_expert_request_typed(
    ExpertWorkspaceCudaV2* ws,
    const ExpertDeviceStorageV2& storage,
    const common::InferRequestMsg& req,
    common::InferResponseMsg* resp) {
    if (ws == nullptr || resp == nullptr) return false;
    if (req.batch_size <= 0 || req.hidden_dim <= 0) return false;

    const size_t elems =
        static_cast<size_t>(req.batch_size) *
        static_cast<size_t>(req.hidden_dim);
    const size_t in_bytes = elems * sizeof(TIn);
    const size_t out_bytes = elems * sizeof(TOut);

    if (req.activation.size() != in_bytes) {
        return false;
    }

    TIn* d_input = nullptr;
    TOut* d_output = nullptr;

    auto cleanup = [&]() {
        if (d_input != nullptr) cudaFree(d_input);
        if (d_output != nullptr) cudaFree(d_output);
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
        nullptr);
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

BackendWorkspaceCudaV2::BackendWorkspaceCudaV2(int local_gpu_id)
    : local_gpu_id_(local_gpu_id) {
    if (local_gpu_id_ < 0) {
        return;
    }

    cudaError_t err = cudaSetDevice(local_gpu_id_);
    if (err != cudaSuccess) {
        local_gpu_id_ = -1;
        return;
    }

    ExpertWorkspaceConfigV2 cfg{};
    cfg.hidden_dim = 7168;
    cfg.inter_dim = 2048;

    ok_ = InitExpertWorkspaceCudaV2(cfg, &ws_);
}

BackendWorkspaceCudaV2::~BackendWorkspaceCudaV2() {
    if (ok_ && local_gpu_id_ >= 0) {
        cudaSetDevice(local_gpu_id_);
        FreeExpertWorkspaceCudaV2(&ws_);
    }
}

bool BackendWorkspaceCudaV2::RunExpertRequest(
    const ExpertDeviceStorageV2& storage,
    const common::InferRequestMsg& req,
    common::InferResponseMsg* resp) {
    if (!ok_ || resp == nullptr) return false;
    if (local_gpu_id_ < 0) return false;
    if (req.batch_size != 1 || req.hidden_dim != 7168) return false;

    const cudaError_t err = cudaSetDevice(local_gpu_id_);
    if (err != cudaSuccess) {
        return false;
    }

    if (req.input_dtype == common::ActivationDType::FP16 &&
        req.output_dtype == common::ActivationDType::FP16) {
        return run_expert_request_typed<__half, __half>(&ws_, storage, req, resp);
    }

#if EXPERT_NODE_V2_HAS_CUDA_BF16
    if (req.input_dtype == common::ActivationDType::FP16 &&
        req.output_dtype == common::ActivationDType::BF16) {
        return run_expert_request_typed<__half, __nv_bfloat16>(&ws_, storage, req, resp);
    }

    if (req.input_dtype == common::ActivationDType::BF16 &&
        req.output_dtype == common::ActivationDType::FP16) {
        return run_expert_request_typed<__nv_bfloat16, __half>(&ws_, storage, req, resp);
    }

    if (req.input_dtype == common::ActivationDType::BF16 &&
        req.output_dtype == common::ActivationDType::BF16) {
        return run_expert_request_typed<__nv_bfloat16, __nv_bfloat16>(&ws_, storage, req, resp);
    }
#endif

    return false;
}

}  // namespace expert_node_v2
