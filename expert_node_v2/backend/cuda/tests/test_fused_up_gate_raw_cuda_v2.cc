#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/backend/activation_codec_v2.h"
#include "expert_node_v2/backend/cuda/backend_cuda_v2.h"
#include "expert_node_v2/backend/dummy_expert_data_v2.h"
#include "expert_node_v2/backend/fp8_lut_v2.h"
#include "expert_node_v2/expert_format_v2.h"

template <class TAct>
bool LaunchFusedUpGateRawCudaV2(
    const MatrixBlockScaleViewV2& w_up,
    const MatrixBlockScaleViewV2& w_gate,
    const TAct* d_x,
    float* d_up_out,
    float* d_gate_out,
    cudaStream_t stream);

template <class TAct>
bool LaunchFusedUpGateRawDoubleAccCudaV2(
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

            up_sum +=
                lut_up[w_up.weight.data[up_w_idx]] *
                w_up.scale.data[up_s_idx] *
                x_val;

            gate_sum +=
                lut_gate[w_gate.weight.data[gate_w_idx]] *
                w_gate.scale.data[gate_s_idx] *
                x_val;
        }

        (*up_out)[static_cast<std::size_t>(row)] = up_sum;
        (*gate_out)[static_cast<std::size_t>(row)] = gate_sum;
    }
}

static void run_fused_up_gate_raw_cpu_warp_order_reference(
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

        float up_lane_sum[32];
        float gate_lane_sum[32];
        for (int lane = 0; lane < 32; ++lane) {
            up_lane_sum[lane] = 0.0f;
            gate_lane_sum[lane] = 0.0f;
        }

        for (int lane = 0; lane < 32; ++lane) {
            for (int k = lane; k < cols; k += 32) {
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

                up_lane_sum[lane] +=
                    lut_up[w_up.weight.data[up_w_idx]] *
                    w_up.scale.data[up_s_idx] *
                    x_val;

                gate_lane_sum[lane] +=
                    lut_gate[w_gate.weight.data[gate_w_idx]] *
                    w_gate.scale.data[gate_s_idx] *
                    x_val;
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            for (int lane = 0; lane < 32 - offset; ++lane) {
                up_lane_sum[lane] += up_lane_sum[lane + offset];
                gate_lane_sum[lane] += gate_lane_sum[lane + offset];
            }
        }

        (*up_out)[static_cast<std::size_t>(row)] = up_lane_sum[0];
        (*gate_out)[static_cast<std::size_t>(row)] = gate_lane_sum[0];
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
    float cos =
        (nr > 0.0f && ng > 0.0f)
            ? (dot / (std::sqrt(nr) * std::sqrt(ng)))
            : 0.0f;
    cos = std::min(1.0f, std::max(-1.0f, cos));

    std::printf("%s: max_abs=%g mean_abs=%g cos=%g\n", name, max_abs, mean_abs, cos);
    for (int i = 0; i < 8; ++i) {
        std::printf("%s ref[%d]=%g gpu[%d]=%g\n", name, i, ref[i], i, gpu[i]);
    }
}

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
        bundle.debug_print("  ");
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
        x_half[static_cast<std::size_t>(i)] = __float2half(x_float[static_cast<std::size_t>(i)]);
    }

    std::vector<float> up_ref_serial, gate_ref_serial;
    run_fused_up_gate_raw_cpu_reference(
        host_view.w_up,
        host_view.w_gate,
        x_fp16.data(),
        common::ActivationDType::FP16,
        &up_ref_serial,
        &gate_ref_serial);

    std::vector<float> up_ref_warp, gate_ref_warp;
    run_fused_up_gate_raw_cpu_warp_order_reference(
        host_view.w_up,
        host_view.w_gate,
        x_fp16.data(),
        common::ActivationDType::FP16,
        &up_ref_warp,
        &gate_ref_warp);

    __half* d_x = nullptr;
    float* d_up_floatacc = nullptr;
    float* d_gate_floatacc = nullptr;
    float* d_up_doubleacc = nullptr;
    float* d_gate_doubleacc = nullptr;

    err = cudaMalloc(reinterpret_cast<void**>(&d_x),
                     static_cast<std::size_t>(hidden_dim) * sizeof(__half));
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_x failed: %s\n", cudaGetErrorString(err));
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_up_floatacc),
                     static_cast<std::size_t>(inter_dim) * sizeof(float));
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_up_floatacc failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_gate_floatacc),
                     static_cast<std::size_t>(inter_dim) * sizeof(float));
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_gate_floatacc failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
        cudaFree(d_up_floatacc);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_up_doubleacc),
                     static_cast<std::size_t>(inter_dim) * sizeof(float));
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_up_doubleacc failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
        cudaFree(d_up_floatacc);
        cudaFree(d_gate_floatacc);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_gate_doubleacc),
                     static_cast<std::size_t>(inter_dim) * sizeof(float));
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_gate_doubleacc failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
        cudaFree(d_up_floatacc);
        cudaFree(d_gate_floatacc);
        cudaFree(d_up_doubleacc);
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
        cudaFree(d_up_floatacc);
        cudaFree(d_gate_floatacc);
        cudaFree(d_up_doubleacc);
        cudaFree(d_gate_doubleacc);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    if (!LaunchFusedUpGateRawCudaV2<__half>(
            storage.view().w_up,
            storage.view().w_gate,
            d_x,
            d_up_floatacc,
            d_gate_floatacc,
            nullptr)) {
        std::printf("LaunchFusedUpGateRawCudaV2 failed\n");
        return 1;
    }

    if (!LaunchFusedUpGateRawDoubleAccCudaV2<__half>(
            storage.view().w_up,
            storage.view().w_gate,
            d_x,
            d_up_doubleacc,
            d_gate_doubleacc,
            nullptr)) {
        std::printf("LaunchFusedUpGateRawDoubleAccCudaV2 failed\n");
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<float> up_gpu_floatacc(inter_dim);
    std::vector<float> gate_gpu_floatacc(inter_dim);
    std::vector<float> up_gpu_doubleacc(inter_dim);
    std::vector<float> gate_gpu_doubleacc(inter_dim);

    cudaMemcpy(
        up_gpu_floatacc.data(),
        d_up_floatacc,
        static_cast<std::size_t>(inter_dim) * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        gate_gpu_floatacc.data(),
        d_gate_floatacc,
        static_cast<std::size_t>(inter_dim) * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        up_gpu_doubleacc.data(),
        d_up_doubleacc,
        static_cast<std::size_t>(inter_dim) * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        gate_gpu_doubleacc.data(),
        d_gate_doubleacc,
        static_cast<std::size_t>(inter_dim) * sizeof(float),
        cudaMemcpyDeviceToHost);

    print_compare("up_sum_cpu_serial_vs_gpu_floatacc", up_ref_serial, up_gpu_floatacc);
    print_compare("up_sum_cpu_warp_vs_gpu_floatacc", up_ref_warp, up_gpu_floatacc);
    print_compare("gate_sum_cpu_serial_vs_gpu_floatacc", gate_ref_serial, gate_gpu_floatacc);
    print_compare("gate_sum_cpu_warp_vs_gpu_floatacc", gate_ref_warp, gate_gpu_floatacc);

    print_compare("up_sum_cpu_serial_vs_gpu_doubleacc", up_ref_serial, up_gpu_doubleacc);
    print_compare("up_sum_cpu_warp_vs_gpu_doubleacc", up_ref_warp, up_gpu_doubleacc);
    print_compare("gate_sum_cpu_serial_vs_gpu_doubleacc", gate_ref_serial, gate_gpu_doubleacc);
    print_compare("gate_sum_cpu_warp_vs_gpu_doubleacc", gate_ref_warp, gate_gpu_doubleacc);

    cudaFree(d_x);
    cudaFree(d_up_floatacc);
    cudaFree(d_gate_floatacc);
    cudaFree(d_up_doubleacc);
    cudaFree(d_gate_doubleacc);
    FreeExpertWeightsCudaV2(&storage);
    return 0;
}
