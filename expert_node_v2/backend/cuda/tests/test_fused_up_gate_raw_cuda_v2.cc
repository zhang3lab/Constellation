#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/backend/activation_codec_v2.h"
#include "expert_node_v2/backend/dummy_expert_data_v2.h"
#include "expert_node_v2/backend/fp8_lut_v2.h"
#include "expert_node_v2/backend/cuda/backend_cuda_v2.h"
#include "expert_node_v2/expert_format_v2.h"

template <class TAct>
bool LaunchFusedUpGateRawCudaV2(
    const MatrixBlockScaleViewV2& w_up,
    const MatrixBlockScaleViewV2& w_gate,
    const TAct* d_x,
    float* d_up_out,
    float* d_gate_out,
    cudaStream_t stream);

static void run_fused_up_gate_raw_cpu_reference(
    const MatrixBlockScaleViewV2& w_up,
    const MatrixBlockScaleViewV2& w_gate,
    const std::uint16_t* x_u16,
    common::ActivationDType input_dtype,
    std::vector<float>* up_out,
    std::vector<float>* gate_out) {
    const int rows = w_up.matrix.rows;
    const int cols = w_up.matrix.cols;

    const float* lut_up = GetHostFp8LutV2(w_up.matrix.fp8_format);
    const float* lut_gate = GetHostFp8LutV2(w_gate.matrix.fp8_format);

    up_out->assign(static_cast<std::size_t>(rows), 0.0f);
    gate_out->assign(static_cast<std::size_t>(rows), 0.0f);

    for (int row = 0; row < rows; ++row) {
        const int rb_up = row / w_up.scale_meta.row_block;
        const int rb_gate = row / w_gate.scale_meta.row_block;

        float up_sum = 0.0f;
        float gate_sum = 0.0f;

        for (int k = 0; k < cols; ++k) {
            const int cb_up = k / w_up.scale_meta.col_block;
            const int cb_gate = k / w_gate.scale_meta.col_block;

            const std::size_t up_w_idx =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(k);
            const std::size_t up_s_idx =
                static_cast<std::size_t>(rb_up) *
                    static_cast<std::size_t>(w_up.scale_meta.num_col_blocks) +
                static_cast<std::size_t>(cb_up);

            const std::size_t gate_w_idx =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(k);
            const std::size_t gate_s_idx =
                static_cast<std::size_t>(rb_gate) *
                    static_cast<std::size_t>(w_gate.scale_meta.num_col_blocks) +
                static_cast<std::size_t>(cb_gate);

            const float x_val = DecodeActivationToFloatV2(input_dtype, x_u16[k]);
            up_sum += lut_up[w_up.weight.data[up_w_idx]] * w_up.scale.data[up_s_idx] * x_val;
            gate_sum += lut_gate[w_gate.weight.data[gate_w_idx]] * w_gate.scale.data[gate_s_idx] * x_val;
        }

        (*up_out)[row] = up_sum;
        (*gate_out)[row] = gate_sum;
    }
}

static void print_compare(
    const char* name,
    const std::vector<float>& ref,
    const std::vector<float>& gpu) {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float dot = 0.0f;
    float nr = 0.0f;
    float ng = 0.0f;

    for (std::size_t i = 0; i < ref.size(); ++i) {
        const float e = std::fabs(ref[i] - gpu[i]);
        if (e > max_abs) max_abs = e;
        mean_abs += e;
        dot += ref[i] * gpu[i];
        nr += ref[i] * ref[i];
        ng += gpu[i] * gpu[i];
    }

    mean_abs /= static_cast<float>(ref.size());
    float cos = (nr > 0.0f && ng > 0.0f) ? (dot / (std::sqrt(nr) * std::sqrt(ng))) : 0.0f;
    if (cos > 1.0f) cos = 1.0f;
    if (cos < -1.0f) cos = -1.0f;

    std::printf("%s: max_abs=%g mean_abs=%g cos=%g\n", name, max_abs, mean_abs, cos);
    for (int i = 0; i < 8; ++i) {
        std::printf("%s ref[%d]=%g gpu[%d]=%g\n", name, i, ref[i], i, gpu[i]);
    }
}

int main() {
    if (cudaSetDevice(0) != cudaSuccess) {
        std::printf("cudaSetDevice failed\n");
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

    std::vector<float> up_ref, gate_ref;
    run_fused_up_gate_raw_cpu_reference(
        host_view.w_up,
        host_view.w_gate,
        x_fp16.data(),
        common::ActivationDType::FP16,
        &up_ref,
        &gate_ref);

    __half* d_x = nullptr;
    float* d_up = nullptr;
    float* d_gate = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_x), hidden_dim * sizeof(__half));
    cudaMalloc(reinterpret_cast<void**>(&d_up), inter_dim * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_gate), inter_dim * sizeof(float));
    cudaMemcpy(d_x, x_half.data(), hidden_dim * sizeof(__half), cudaMemcpyHostToDevice);

    if (!LaunchFusedUpGateRawCudaV2<__half>(
            storage.view().w_up,
            storage.view().w_gate,
            d_x,
            d_up,
            d_gate,
            nullptr)) {
        std::printf("LaunchFusedUpGateRawCudaV2 failed\n");
        return 1;
    }
    cudaDeviceSynchronize();

    std::vector<float> up_gpu(inter_dim), gate_gpu(inter_dim);
    cudaMemcpy(up_gpu.data(), d_up, inter_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gate_gpu.data(), d_gate, inter_dim * sizeof(float), cudaMemcpyDeviceToHost);

    print_compare("up_sum", up_ref, up_gpu);
    print_compare("gate_sum", gate_ref, gate_gpu);

    cudaFree(d_x);
    cudaFree(d_up);
    cudaFree(d_gate);
    FreeExpertWeightsCudaV2(&storage);
    return 0;
}
