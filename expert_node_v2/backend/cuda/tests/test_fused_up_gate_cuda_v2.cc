#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/backend/cuda/backend_cuda_v2.h"
#include "expert_node_v2/backend/cuda/fused_up_gate_cuda_v2.h"
#include "expert_node_v2/backend/dummy_expert_data_v2.h"
#include "expert_node_v2/backend/expert_reference_v2.h"
#include "expert_node_v2/expert_format_v2.h"

int main() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::printf("cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    ExpertTensorBundleV2 bundle;
    FillDummyExpertBundleV2(&bundle);

    ExpertWeightsViewV2 host_view;
    if (!BuildExpertWeightsViewV2(bundle, &host_view)) {
        std::printf("BuildExpertWeightsViewV2 failed\n");
        return 1;
    }

    ExpertDeviceStorageV2 storage;
    if (!UploadExpertCudaV2(0, bundle, &storage)) {
        std::printf("UploadExpertCudaV2 failed\n");
        return 1;
    }

    const int hidden_dim = 7168;
    const int inter_dim = 2048;

    std::vector<float> x_float;
    std::vector<std::uint16_t> x_fp16;
    FillDummyInputActivationV2(
        hidden_dim,
        common::ActivationDType::FP16,
        &x_float,
        &x_fp16);

    std::vector<__half> x_half(hidden_dim);
    for (int i = 0; i < hidden_dim; ++i) {
        x_half[i] = __float2half(x_float[i]);
    }

    std::vector<float> h_ref;
    if (!RunFusedUpGateReferenceV2(
            host_view.w_up,
            host_view.w_gate,
            x_fp16.data(),
            common::ActivationDType::FP16,
            &h_ref)) {
        std::printf("RunFusedUpGateReferenceV2 failed\n");
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    __half* d_x = nullptr;
    float* d_h = nullptr;

    err = cudaMalloc(reinterpret_cast<void**>(&d_x),
                     static_cast<std::size_t>(hidden_dim) * sizeof(__half));
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_x failed: %s\n", cudaGetErrorString(err));
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_h),
                     static_cast<std::size_t>(inter_dim) * sizeof(float));
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_h failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    err = cudaMemcpy(
        d_x,
        x_half.data(),
        static_cast<std::size_t>(hidden_dim) * sizeof(__half),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::printf("cudaMemcpy d_x failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
        cudaFree(d_h);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    if (!LaunchFusedUpGateCudaV2Impl<__half>(
            storage.view().w_up,
            storage.view().w_gate,
            d_x,
            d_h,
            nullptr)) {
        std::printf("LaunchFusedUpGateCudaV2Impl<__half> failed\n");
        cudaFree(d_x);
        cudaFree(d_h);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
        cudaFree(d_h);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    std::vector<float> h_gpu(inter_dim);
    err = cudaMemcpy(
        h_gpu.data(),
        d_h,
        static_cast<std::size_t>(inter_dim) * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::printf("cudaMemcpy h_gpu failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
        cudaFree(d_h);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    float dot = 0.0f;
    float norm_ref = 0.0f;
    float norm_gpu = 0.0f;

    for (int i = 0; i < inter_dim; ++i) {
        const float ref_v = h_ref[i];
        const float gpu_v = h_gpu[i];

        const float abs_err = std::fabs(ref_v - gpu_v);
        if (abs_err > max_abs) max_abs = abs_err;
        sum_abs += abs_err;

        dot += ref_v * gpu_v;
        norm_ref += ref_v * ref_v;
        norm_gpu += gpu_v * gpu_v;
    }

    const float mean_abs = sum_abs / static_cast<float>(inter_dim);
    const float cos =
        (norm_ref > 0.0f && norm_gpu > 0.0f)
            ? (dot / (std::sqrt(norm_ref) * std::sqrt(norm_gpu)))
            : 0.0f;

    std::printf(
        "fused up/gate compare: max_abs=%g mean_abs=%g cos=%g\n",
        max_abs, mean_abs, cos);

    for (int i = 0; i < 8; ++i) {
        std::printf("ref[%d]=%g gpu[%d]=%g\n",
                    i, h_ref[i], i, h_gpu[i]);
    }

    cudaFree(d_x);
    cudaFree(d_h);
    FreeExpertWeightsCudaV2(&storage);

    const bool pass = (max_abs <= 1e-3f) && (cos >= 0.99999f);
    std::printf("PASS=%d\n", pass ? 1 : 0);
    return pass ? 0 : 1;
}
