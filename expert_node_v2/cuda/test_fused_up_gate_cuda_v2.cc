#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "expert_node_v2/expert_format_v2.h"
#include "expert_node_v2/cuda/backend_cuda_v2.h"
#include "expert_node_v2/cuda/fused_up_gate_cuda_v2.h"

static float decode_torch_e4m3fn_byte(std::uint8_t v) {
    const int sign = (v >> 7) & 0x1;
    const int exp = (v >> 3) & 0xF;
    const int mant = v & 0x7;
    const float s = sign ? -1.0f : 1.0f;
    const int bias = 7;

    if (exp == 0) {
        if (mant == 0) return s * 0.0f;
        return s * std::ldexp(static_cast<float>(mant) / 8.0f, 1 - bias);
    }
    if (exp == 0xF) {
        if (mant == 0x7) return s * 448.0f;
        return s * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - bias);
    }
    return s * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - bias);
}

static void fill_dummy_bundle(ExpertTensorBundleV2* bundle) {
    const int inter_dim = 2048;
    const int hidden_dim = 7168;
    const int row_block = 128;
    const int col_block = 128;

    const int up_num_row_blocks = (inter_dim + row_block - 1) / row_block;
    const int up_num_col_blocks = (hidden_dim + col_block - 1) / col_block;

    bundle->w_up.shape = {inter_dim, hidden_dim};
    bundle->w_up.dtype = "torch.float8_e4m3fn";
    bundle->w_up.bytes.resize(static_cast<std::size_t>(inter_dim) * hidden_dim);
    bundle->w_up.ready = true;
    for (std::size_t i = 0; i < bundle->w_up.bytes.size(); ++i) {
        bundle->w_up.bytes[i] = static_cast<std::uint8_t>((i * 7 + 3) & 0xff);
    }

    bundle->w_up_scale.shape = {up_num_row_blocks, up_num_col_blocks};
    bundle->w_up_scale.dtype = "torch.float32";
    bundle->w_up_scale.bytes.resize(
        static_cast<std::size_t>(up_num_row_blocks) *
        static_cast<std::size_t>(up_num_col_blocks) * sizeof(float));
    bundle->w_up_scale.ready = true;
    {
        float* scales = reinterpret_cast<float*>(bundle->w_up_scale.bytes.data());
        for (int rb = 0; rb < up_num_row_blocks; ++rb) {
            for (int cb = 0; cb < up_num_col_blocks; ++cb) {
                scales[rb * up_num_col_blocks + cb] =
                    0.010f + 0.0005f * static_cast<float>(rb) +
                    0.00005f * static_cast<float>(cb);
            }
        }
    }

    bundle->w_gate.shape = {inter_dim, hidden_dim};
    bundle->w_gate.dtype = "torch.float8_e4m3fn";
    bundle->w_gate.bytes.resize(static_cast<std::size_t>(inter_dim) * hidden_dim);
    bundle->w_gate.ready = true;
    for (std::size_t i = 0; i < bundle->w_gate.bytes.size(); ++i) {
        bundle->w_gate.bytes[i] = static_cast<std::uint8_t>((i * 11 + 5) & 0xff);
    }

    bundle->w_gate_scale.shape = {up_num_row_blocks, up_num_col_blocks};
    bundle->w_gate_scale.dtype = "torch.float32";
    bundle->w_gate_scale.bytes.resize(
        static_cast<std::size_t>(up_num_row_blocks) *
        static_cast<std::size_t>(up_num_col_blocks) * sizeof(float));
    bundle->w_gate_scale.ready = true;
    {
        float* scales = reinterpret_cast<float*>(bundle->w_gate_scale.bytes.data());
        for (int rb = 0; rb < up_num_row_blocks; ++rb) {
            for (int cb = 0; cb < up_num_col_blocks; ++cb) {
                scales[rb * up_num_col_blocks + cb] =
                    0.020f + 0.0003f * static_cast<float>(rb) +
                    0.00007f * static_cast<float>(cb);
            }
        }
    }

    const int down_rows = 7168;
    const int down_cols = 2048;
    const int down_num_row_blocks = (down_rows + row_block - 1) / row_block;
    const int down_num_col_blocks = (down_cols + col_block - 1) / col_block;

    bundle->w_down.shape = {down_rows, down_cols};
    bundle->w_down.dtype = "torch.float8_e4m3fn";
    bundle->w_down.bytes.resize(static_cast<std::size_t>(down_rows) * down_cols, 0);
    bundle->w_down.ready = true;

    bundle->w_down_scale.shape = {down_num_row_blocks, down_num_col_blocks};
    bundle->w_down_scale.dtype = "torch.float32";
    bundle->w_down_scale.bytes.resize(
        static_cast<std::size_t>(down_num_row_blocks) *
        static_cast<std::size_t>(down_num_col_blocks) * sizeof(float), 0);
    bundle->w_down_scale.ready = true;
}

static std::vector<float> run_fused_up_gate_cpu_reference(
    const ExpertTensorBundleV2& bundle,
    const std::vector<float>& x_host) {
    const int inter_dim = 2048;
    const int hidden_dim = 7168;
    const int row_block = 128;
    const int col_block = 128;
    const int num_col_blocks = (hidden_dim + col_block - 1) / col_block;

    const std::uint8_t* up_weights = bundle.w_up.bytes.data();
    const float* up_scales =
        reinterpret_cast<const float*>(bundle.w_up_scale.bytes.data());

    const std::uint8_t* gate_weights = bundle.w_gate.bytes.data();
    const float* gate_scales =
        reinterpret_cast<const float*>(bundle.w_gate_scale.bytes.data());

    std::vector<float> h(inter_dim, 0.0f);

    for (int r = 0; r < inter_dim; ++r) {
        const int rb = r / row_block;
        float up_sum = 0.0f;
        float gate_sum = 0.0f;

        for (int k = 0; k < hidden_dim; ++k) {
            const int cb = k / col_block;

            const std::size_t w_idx =
                static_cast<std::size_t>(r) * static_cast<std::size_t>(hidden_dim) +
                static_cast<std::size_t>(k);
            const std::size_t s_idx =
                static_cast<std::size_t>(rb) *
                    static_cast<std::size_t>(num_col_blocks) +
                static_cast<std::size_t>(cb);

            up_sum += decode_torch_e4m3fn_byte(up_weights[w_idx]) * up_scales[s_idx] * x_host[k];
            gate_sum += decode_torch_e4m3fn_byte(gate_weights[w_idx]) * gate_scales[s_idx] * x_host[k];
        }

        const float silu_gate = gate_sum / (1.0f + std::exp(-gate_sum));
        h[r] = silu_gate * up_sum;
    }

    return h;
}

int main() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::printf("cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    ExpertTensorBundleV2 bundle;
    fill_dummy_bundle(&bundle);

    ExpertDeviceStorageV2 storage;
    if (!UploadExpertCudaV2(bundle, &storage)) {
        std::printf("UploadExpertCudaV2 failed\n");
        return 1;
    }

    const int hidden_dim = 7168;
    const int inter_dim = 2048;

    std::vector<float> x_float(hidden_dim);
    std::vector<__half> x_half(hidden_dim);
    std::vector<float> x_cpu_quant(hidden_dim);

    for (int i = 0; i < hidden_dim; ++i) {
        x_float[i] = std::sin(0.0005f * static_cast<float>(i));
        x_half[i] = __float2half(x_float[i]);
        x_cpu_quant[i] = __half2float(x_half[i]);
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

    std::vector<float> h_cpu_full = run_fused_up_gate_cpu_reference(bundle, x_float);
    std::vector<float> h_cpu_quant = run_fused_up_gate_cpu_reference(bundle, x_cpu_quant);

    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    float dot = 0.0f;
    float norm_cpu = 0.0f;
    float norm_gpu = 0.0f;

    float max_abs_full = 0.0f;
    float sum_abs_full = 0.0f;

    for (int i = 0; i < inter_dim; ++i) {
        const float cpu_full_v = h_cpu_full[i];
        const float cpu_q_v = h_cpu_quant[i];
        const float gpu_v = h_gpu[i];

        const float abs_err = std::fabs(cpu_q_v - gpu_v);
        const float abs_err_full = std::fabs(cpu_full_v - gpu_v);

        if (abs_err > max_abs) max_abs = abs_err;
        if (abs_err_full > max_abs_full) max_abs_full = abs_err_full;

        sum_abs += abs_err;
        sum_abs_full += abs_err_full;

        dot += cpu_q_v * gpu_v;
        norm_cpu += cpu_q_v * cpu_q_v;
        norm_gpu += gpu_v * gpu_v;
    }

    const float mean_abs = sum_abs / static_cast<float>(inter_dim);
    const float mean_abs_full = sum_abs_full / static_cast<float>(inter_dim);
    const float cos =
        (norm_cpu > 0.0f && norm_gpu > 0.0f)
            ? (dot / (std::sqrt(norm_cpu) * std::sqrt(norm_gpu)))
            : 0.0f;

    std::printf(
        "fused up/gate compare: "
        "q_max_abs=%g q_mean_abs=%g full_max_abs=%g full_mean_abs=%g cos=%g\n",
        max_abs, mean_abs, max_abs_full, mean_abs_full, cos);

    for (int i = 0; i < 8; ++i) {
        std::printf("cpu_full[%d]=%g cpu_q[%d]=%g gpu[%d]=%g\n",
                    i, h_cpu_full[i], i, h_cpu_quant[i], i, h_gpu[i]);
    }

    cudaFree(d_x);
    cudaFree(d_h);
    FreeExpertWeightsCudaV2(&storage);
    return 0;
}
