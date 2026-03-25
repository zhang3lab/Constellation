#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "expert_node_v2/expert_format_v2.h"
#include "expert_node_v2/cuda/backend_cuda_v2.h"
#include "expert_node_v2/cuda/down_cuda_v2.h"

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

    // finite-only torch.float8_e4m3fn
    if (exp == 0xF) {
        if (mant == 0x7) return s * 448.0f;
        return s * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - bias);
    }

    return s * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - bias);
}

static void fill_dummy_down_bundle(ExpertTensorBundleV2* bundle) {
    const int down_rows = 7168;
    const int down_cols = 2048;
    const int row_block = 128;
    const int col_block = 128;
    const int down_num_row_blocks = (down_rows + row_block - 1) / row_block;
    const int down_num_col_blocks = (down_cols + col_block - 1) / col_block;

    bundle->w_down.shape = {down_rows, down_cols};
    bundle->w_down.dtype = "torch.float8_e4m3fn";
    bundle->w_down.bytes.resize(
        static_cast<std::size_t>(down_rows) * static_cast<std::size_t>(down_cols));
    bundle->w_down.ready = true;

    for (std::size_t i = 0; i < bundle->w_down.bytes.size(); ++i) {
        bundle->w_down.bytes[i] = static_cast<std::uint8_t>((i * 13 + 17) & 0xff);
    }

    bundle->w_down_scale.shape = {down_num_row_blocks, down_num_col_blocks};
    bundle->w_down_scale.dtype = "torch.float32";
    bundle->w_down_scale.bytes.resize(
        static_cast<std::size_t>(down_num_row_blocks) *
        static_cast<std::size_t>(down_num_col_blocks) *
        sizeof(float));
    bundle->w_down_scale.ready = true;

    float* down_scales = reinterpret_cast<float*>(bundle->w_down_scale.bytes.data());
    for (int rb = 0; rb < down_num_row_blocks; ++rb) {
        for (int cb = 0; cb < down_num_col_blocks; ++cb) {
            const int idx = rb * down_num_col_blocks + cb;
            down_scales[idx] = 0.005f + 0.001f * static_cast<float>(rb) +
                               0.0001f * static_cast<float>(cb);
        }
    }

    const int up_rows = 2048;
    const int up_cols = 7168;
    const int up_num_row_blocks = (up_rows + row_block - 1) / row_block;
    const int up_num_col_blocks = (up_cols + col_block - 1) / col_block;

    bundle->w_up.shape = {up_rows, up_cols};
    bundle->w_up.dtype = "torch.float8_e4m3fn";
    bundle->w_up.bytes.resize(
        static_cast<std::size_t>(up_rows) * static_cast<std::size_t>(up_cols));
    bundle->w_up.ready = true;
    for (std::size_t i = 0; i < bundle->w_up.bytes.size(); ++i) {
        bundle->w_up.bytes[i] = static_cast<std::uint8_t>((i * 7 + 3) & 0xff);
    }

    bundle->w_up_scale.shape = {up_num_row_blocks, up_num_col_blocks};
    bundle->w_up_scale.dtype = "torch.float32";
    bundle->w_up_scale.bytes.resize(
        static_cast<std::size_t>(up_num_row_blocks) *
        static_cast<std::size_t>(up_num_col_blocks) *
        sizeof(float));
    bundle->w_up_scale.ready = true;
    float* up_scales = reinterpret_cast<float*>(bundle->w_up_scale.bytes.data());
    for (int rb = 0; rb < up_num_row_blocks; ++rb) {
        for (int cb = 0; cb < up_num_col_blocks; ++cb) {
            const int idx = rb * up_num_col_blocks + cb;
            up_scales[idx] = 0.010f + 0.0005f * static_cast<float>(rb) +
                             0.00005f * static_cast<float>(cb);
        }
    }

    bundle->w_gate.shape = {up_rows, up_cols};
    bundle->w_gate.dtype = "torch.float8_e4m3fn";
    bundle->w_gate.bytes.resize(
        static_cast<std::size_t>(up_rows) * static_cast<std::size_t>(up_cols));
    bundle->w_gate.ready = true;
    for (std::size_t i = 0; i < bundle->w_gate.bytes.size(); ++i) {
        bundle->w_gate.bytes[i] = static_cast<std::uint8_t>((i * 11 + 5) & 0xff);
    }

    bundle->w_gate_scale.shape = {up_num_row_blocks, up_num_col_blocks};
    bundle->w_gate_scale.dtype = "torch.float32";
    bundle->w_gate_scale.bytes.resize(
        static_cast<std::size_t>(up_num_row_blocks) *
        static_cast<std::size_t>(up_num_col_blocks) *
        sizeof(float));
    bundle->w_gate_scale.ready = true;
    float* gate_scales = reinterpret_cast<float*>(bundle->w_gate_scale.bytes.data());
    for (int rb = 0; rb < up_num_row_blocks; ++rb) {
        for (int cb = 0; cb < up_num_col_blocks; ++cb) {
            const int idx = rb * up_num_col_blocks + cb;
            gate_scales[idx] = 0.020f + 0.0003f * static_cast<float>(rb) +
                               0.00007f * static_cast<float>(cb);
        }
    }
}

static std::vector<float> run_down_cpu_reference(
    const ExpertTensorBundleV2& bundle,
    const std::vector<float>& h_host) {
    const int rows = 7168;
    const int cols = 2048;
    const int row_block = 128;
    const int col_block = 128;
    const int num_col_blocks = (cols + col_block - 1) / col_block;

    const std::uint8_t* weights = bundle.w_down.bytes.data();
    const float* scales =
        reinterpret_cast<const float*>(bundle.w_down_scale.bytes.data());

    std::vector<float> y(rows, 0.0f);

    for (int r = 0; r < rows; ++r) {
        const int rb = r / row_block;
        float sum = 0.0f;

        for (int k = 0; k < cols; ++k) {
            const int cb = k / col_block;

            const std::size_t w_idx =
                static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(k);
            const std::size_t s_idx =
                static_cast<std::size_t>(rb) *
                    static_cast<std::size_t>(num_col_blocks) +
                static_cast<std::size_t>(cb);

            const float scale = scales[s_idx];
            const float w = decode_torch_e4m3fn_byte(weights[w_idx]) * scale;
            sum += w * h_host[k];
        }

        y[r] = sum;
    }

    return y;
}

int main() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::printf("cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    ExpertTensorBundleV2 bundle;
    fill_dummy_down_bundle(&bundle);

    ExpertDeviceStorageV2 storage;
    if (!UploadExpertCudaV2(bundle, &storage)) {
        std::printf("UploadExpertCudaV2 failed\n");
        return 1;
    }

    const int inter_dim = 2048;
    const int hidden_dim = 7168;

    std::vector<float> h_host(inter_dim);
    for (int i = 0; i < inter_dim; ++i) {
        h_host[i] = std::sin(0.001f * static_cast<float>(i));
    }

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

    std::vector<__half> y_host(hidden_dim);
    err = cudaMemcpy(
        y_host.data(),
        d_y,
        static_cast<std::size_t>(hidden_dim) * sizeof(__half),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::printf("cudaMemcpy y_host failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_h);
        cudaFree(d_y);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    std::vector<float> y_cpu = run_down_cpu_reference(bundle, h_host);

    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    float dot = 0.0f;
    float norm_cpu = 0.0f;
    float norm_gpu = 0.0f;

    int same_adj = 0;
    for (int i = 1; i < hidden_dim; ++i) {
        const float a = __half2float(y_host[i - 1]);
        const float b = __half2float(y_host[i]);
        if (a == b) same_adj++;
    }

    for (int i = 0; i < hidden_dim; ++i) {
        const float gpu_v = __half2float(y_host[i]);
        const float cpu_v = y_cpu[i];
        const float abs_err = std::fabs(gpu_v - cpu_v);

        if (abs_err > max_abs) max_abs = abs_err;
        sum_abs += abs_err;
        dot += gpu_v * cpu_v;
        norm_cpu += cpu_v * cpu_v;
        norm_gpu += gpu_v * gpu_v;
    }

    const float mean_abs = sum_abs / static_cast<float>(hidden_dim);
    const float cos =
        (norm_cpu > 0.0f && norm_gpu > 0.0f)
            ? (dot / (std::sqrt(norm_cpu) * std::sqrt(norm_gpu)))
            : 0.0f;

    std::printf("compare: max_abs=%g mean_abs=%g cos=%g same_adj=%d/%d\n",
                max_abs, mean_abs, cos, same_adj, hidden_dim - 1);

    for (int i = 0; i < 8; ++i) {
        std::printf("cpu[%d]=%g gpu[%d]=%g\n",
                    i, y_cpu[i], i, __half2float(y_host[i]));
    }

    for (int i = 120; i < 128; ++i) {
        std::printf("cpu[%d]=%g gpu[%d]=%g\n",
                    i, y_cpu[i], i, __half2float(y_host[i]));
    }

    for (int i = 128; i < 136; ++i) {
        std::printf("cpu[%d]=%g gpu[%d]=%g\n",
                    i, y_cpu[i], i, __half2float(y_host[i]));
    }

    cudaFree(d_h);
    cudaFree(d_y);
    FreeExpertWeightsCudaV2(&storage);
    return 0;
}
