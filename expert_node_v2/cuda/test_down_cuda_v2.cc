#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "expert_node_v2/expert_format_v2.h"
#include "expert_node_v2/cuda/backend_cuda_v2.h"
#include "expert_node_v2/cuda/mlp_blockscale_cuda_v2.h"

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

    int finite_count = 0;
    float min_v = 0.0f;
    float max_v = 0.0f;
    float sum_v = 0.0f;

    for (int i = 0; i < hidden_dim; ++i) {
        const float v = __half2float(y_host[i]);
        if (std::isfinite(v)) {
            if (finite_count == 0) {
                min_v = v;
                max_v = v;
            } else {
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }
            sum_v += v;
            finite_count++;
        }
    }

    std::printf("down test: finite=%d/%d min=%g max=%g mean=%g\n",
                finite_count,
                hidden_dim,
                min_v,
                max_v,
                finite_count > 0 ? (sum_v / finite_count) : 0.0f);

    for (int i = 0; i < 8; ++i) {
        std::printf("y[%d] = %g\n", i, __half2float(y_host[i]));
    }

    cudaFree(d_h);
    cudaFree(d_y);
    FreeExpertWeightsCudaV2(&storage);
    return 0;
}
