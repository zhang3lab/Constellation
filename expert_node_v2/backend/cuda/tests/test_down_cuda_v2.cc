#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/backend/activation_codec_v2.h"
#include "expert_node_v2/backend/cuda/backend_cuda_v2.h"
#include "expert_node_v2/backend/cuda/down_cuda_v2.h"
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

    const int inter_dim = 2048;
    const int hidden_dim = 7168;

    std::vector<float> h_host(inter_dim);
    for (int i = 0; i < inter_dim; ++i) {
        h_host[i] = std::sin(0.001f * static_cast<float>(i));
    }

    std::vector<std::uint8_t> y_ref_bytes;
    if (!RunDownReferenceV2(
            host_view.w_down,
            h_host.data(),
            common::ActivationDType::FP16,
            &y_ref_bytes)) {
        std::printf("RunDownReferenceV2 failed\n");
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }
    const auto* y_ref_u16 =
        reinterpret_cast<const std::uint16_t*>(y_ref_bytes.data());

    float* d_h = nullptr;
    __half* d_y = nullptr;

    err = cudaMalloc(reinterpret_cast<void**>(&d_h),
                     static_cast<std::size_t>(inter_dim) * sizeof(float));
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_h failed: %s\n", cudaGetErrorString(err));
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_y),
                     static_cast<std::size_t>(hidden_dim) * sizeof(__half));
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_y failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_h);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    err = cudaMemcpy(
        d_h,
        h_host.data(),
        static_cast<std::size_t>(inter_dim) * sizeof(float),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::printf("cudaMemcpy d_h failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_h);
        cudaFree(d_y);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    if (!LaunchDownCudaV2Impl<__half>(storage.view().w_down, d_h, d_y, nullptr)) {
        std::printf("LaunchDownCudaV2Impl<__half> failed\n");
        cudaFree(d_h);
        cudaFree(d_y);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_h);
        cudaFree(d_y);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    std::vector<__half> y_gpu_half(hidden_dim);
    err = cudaMemcpy(
        y_gpu_half.data(),
        d_y,
        static_cast<std::size_t>(hidden_dim) * sizeof(__half),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::printf("cudaMemcpy y_gpu_half failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_h);
        cudaFree(d_y);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    float dot = 0.0f;
    float norm_ref = 0.0f;
    float norm_gpu = 0.0f;

    int same_adj = 0;
    for (int i = 1; i < hidden_dim; ++i) {
        const float a = __half2float(y_gpu_half[i - 1]);
        const float b = __half2float(y_gpu_half[i]);
        if (a == b) same_adj++;
    }

    for (int i = 0; i < hidden_dim; ++i) {
        const float gpu_v = __half2float(y_gpu_half[i]);
        const float ref_v = DecodeActivationToFloatV2(
            common::ActivationDType::FP16, y_ref_u16[i]);
        const float abs_err = std::fabs(gpu_v - ref_v);

        if (abs_err > max_abs) max_abs = abs_err;
        sum_abs += abs_err;
        dot += gpu_v * ref_v;
        norm_ref += ref_v * ref_v;
        norm_gpu += gpu_v * gpu_v;
    }

    const float mean_abs = sum_abs / static_cast<float>(hidden_dim);
    const float cos =
        (norm_ref > 0.0f && norm_gpu > 0.0f)
            ? (dot / (std::sqrt(norm_ref) * std::sqrt(norm_gpu)))
            : 0.0f;

    std::printf("compare: max_abs=%g mean_abs=%g cos=%g same_adj=%d/%d\n",
                max_abs, mean_abs, cos, same_adj, hidden_dim - 1);

    for (int i = 0; i < 8; ++i) {
        const float ref_v = DecodeActivationToFloatV2(
            common::ActivationDType::FP16, y_ref_u16[i]);
        std::printf("ref[%d]=%g gpu[%d]=%g\n",
                    i, ref_v, i, __half2float(y_gpu_half[i]));
    }

    for (int i = 120; i < 128; ++i) {
        const float ref_v = DecodeActivationToFloatV2(
            common::ActivationDType::FP16, y_ref_u16[i]);
        std::printf("ref[%d]=%g gpu[%d]=%g\n",
                    i, ref_v, i, __half2float(y_gpu_half[i]));
    }

    for (int i = 128; i < 136; ++i) {
        const float ref_v = DecodeActivationToFloatV2(
            common::ActivationDType::FP16, y_ref_u16[i]);
        std::printf("ref[%d]=%g gpu[%d]=%g\n",
                    i, ref_v, i, __half2float(y_gpu_half[i]));
    }

    cudaFree(d_h);
    cudaFree(d_y);
    FreeExpertWeightsCudaV2(&storage);
    return 0;
}
