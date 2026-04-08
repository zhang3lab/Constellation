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

static void verify_fp16_encode_matches_cuda_half(
    const std::vector<float>& x_float) {
    int first_bad = -1;

    for (std::size_t i = 0; i < x_float.size(); ++i) {
        const float x = x_float[i];

        const std::uint16_t cpu_bits = EncodeFloatToFp16V2(x);

        const __half h = __float2half(x);
        std::uint16_t cuda_bits = 0;
        static_assert(sizeof(cuda_bits) == sizeof(h));
        std::memcpy(&cuda_bits, &h, sizeof(cuda_bits));

        if (cpu_bits != cuda_bits) {
            first_bad = static_cast<int>(i);

            const float cpu_decode = DecodeFp16ToFloatV2(cpu_bits);
            const float cuda_decode = __half2float(h);

            std::printf(
                "first fp16 encode mismatch "
                "k=%d x=%g cpu_bits=0x%04x cuda_bits=0x%04x "
                "cpu_decode=%g cuda_decode=%g\n",
                first_bad,
                x,
                static_cast<unsigned>(cpu_bits),
                static_cast<unsigned>(cuda_bits),
                cpu_decode,
                cuda_decode);

            break;
        }
    }

    if (first_bad < 0) {
        std::printf("fp16 encode check: all matched\n");
    }
}

static void print_up_cpu_debug_row0(
    const MatrixBlockScaleViewV2& w_up,
    const std::uint16_t* x_u16,
    common::ActivationDType input_dtype,
    int k_begin,
    int k_end) {
    const int row = 0;
    const int cols = w_up.matrix.cols;

    const float* lut_up = GetHostFp8LutV2(w_up.matrix.fp8_format);
    const int rb_up = row / w_up.scale_meta.row_block;

    std::printf("=== CPU up debug row=0 k=[%d,%d) ===\n", k_begin, k_end);
    for (int k = k_begin; k < cols && k < k_end; ++k) {
        const int cb_up = k / w_up.scale_meta.col_block;

        const std::size_t up_w_idx =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
            static_cast<std::size_t>(k);
        const std::size_t up_s_idx =
            static_cast<std::size_t>(rb_up) *
                static_cast<std::size_t>(w_up.scale_meta.num_col_blocks) +
            static_cast<std::size_t>(cb_up);

        const std::uint8_t w_byte = w_up.weight.data[up_w_idx];
        const float scale = w_up.scale.data[up_s_idx];
        const float decoded = lut_up[w_byte];
        const float x_val = DecodeActivationToFloatV2(input_dtype, x_u16[k]);
        const float contrib = decoded * scale * x_val;

        std::printf(
            "CPU k=%d cb=%d w_idx=%zu s_idx=%zu w_byte=%u scale=%g decoded=%g x=%g contrib=%g\n",
            k, cb_up, up_w_idx, up_s_idx, static_cast<unsigned>(w_byte),
            scale, decoded, x_val, contrib);
    }
}

static void build_up_cpu_row0_contribs(
    const MatrixBlockScaleViewV2& w_up,
    const std::uint16_t* x_u16,
    common::ActivationDType input_dtype,
    std::vector<float>* contribs) {
    const int row = 0;
    const int cols = w_up.matrix.cols;
    const float* lut_up = GetHostFp8LutV2(w_up.matrix.fp8_format);
    const int rb_up = row / w_up.scale_meta.row_block;

    contribs->assign(static_cast<std::size_t>(cols), 0.0f);

    for (int k = 0; k < cols; ++k) {
        const int cb_up = k / w_up.scale_meta.col_block;

        const std::size_t up_w_idx =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
            static_cast<std::size_t>(k);
        const std::size_t up_s_idx =
            static_cast<std::size_t>(rb_up) *
                static_cast<std::size_t>(w_up.scale_meta.num_col_blocks) +
            static_cast<std::size_t>(cb_up);

        const std::uint8_t w_byte = w_up.weight.data[up_w_idx];
        const float scale = w_up.scale.data[up_s_idx];
        const float decoded = lut_up[w_byte];
        const float x_val = DecodeActivationToFloatV2(input_dtype, x_u16[k]);
        (*contribs)[static_cast<std::size_t>(k)] = decoded * scale * x_val;
    }
}

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

struct UpDebugItemCudaV2 {
    int k;
    int cb;
    std::size_t w_idx;
    std::size_t s_idx;
    std::uint8_t w_byte;
    float scale;
    float decoded;
    float x_val;
    float contrib;
};

template <class TAct>
bool LaunchDebugUpRow0CudaV2(
    const MatrixBlockScaleViewV2& w_up,
    const TAct* d_x,
    UpDebugItemCudaV2* d_out,
    int k_limit,
    cudaStream_t stream);

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

    verify_fp16_encode_matches_cuda_half(x_float);
    std::vector<__half> x_half(hidden_dim);
    for (int i = 0; i < hidden_dim; ++i) {
        x_half[static_cast<std::size_t>(i)] = __float2half(x_float[static_cast<std::size_t>(i)]);
    }

    print_up_cpu_debug_row0(
    host_view.w_up,
    x_fp16.data(),
    common::ActivationDType::FP16,
    120,
    136);

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
    const int k_limit = hidden_dim;
UpDebugItemCudaV2* d_debug = nullptr;
cudaMalloc(reinterpret_cast<void**>(&d_debug),
           static_cast<std::size_t>(k_limit) * sizeof(UpDebugItemCudaV2));

if (!LaunchDebugUpRow0CudaV2<__half>(
        storage.view().w_up,
        d_x,
        d_debug,
        k_limit,
        nullptr)) {
    std::printf("LaunchDebugUpRow0CudaV2 failed\n");
    return 1;
}

cudaDeviceSynchronize();

std::vector<UpDebugItemCudaV2> debug_items(static_cast<std::size_t>(k_limit));
cudaMemcpy(
    debug_items.data(),
    d_debug,
    static_cast<std::size_t>(k_limit) * sizeof(UpDebugItemCudaV2),
    cudaMemcpyDeviceToHost);

std::vector<float> up_cpu_contribs;
build_up_cpu_row0_contribs(
    host_view.w_up,
    x_fp16.data(),
    common::ActivationDType::FP16,
    &up_cpu_contribs);

int first_bad = -1;
float max_contrib_abs = 0.0f;
int max_contrib_k = -1;

for (int k = 0; k < hidden_dim; ++k) {
    const float cpu_v = up_cpu_contribs[static_cast<std::size_t>(k)];
    const float gpu_v = debug_items[static_cast<std::size_t>(k)].contrib;
    const float err = std::fabs(cpu_v - gpu_v);

    if (err > max_contrib_abs) {
        max_contrib_abs = err;
        max_contrib_k = k;
    }
    if (first_bad < 0 && err > 1e-12f) {
        first_bad = k;
    }
}

std::printf("up contrib first_bad=%d max_abs=%g max_k=%d\n",
            first_bad, max_contrib_abs, max_contrib_k);

if (first_bad >= 0) {
    const auto& it = debug_items[static_cast<std::size_t>(first_bad)];
    std::printf(
        "FIRST_BAD k=%d cpu_contrib=%g gpu_contrib=%g cb=%d w_idx=%zu s_idx=%zu w_byte=%u scale=%g decoded=%g x=%g\n",
        first_bad,
        up_cpu_contribs[static_cast<std::size_t>(first_bad)],
        it.contrib,
        it.cb,
        it.w_idx,
        it.s_idx,
        static_cast<unsigned>(it.w_byte),
        it.scale,
        it.decoded,
        it.x_val);
}

cudaFree(d_debug);
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
