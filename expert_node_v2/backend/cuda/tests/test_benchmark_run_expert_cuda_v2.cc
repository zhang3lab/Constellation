#include <cuda_fp16.h>
#if EXPERT_NODE_V2_HAS_CUDA_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "expert_node_v2/build_config_v2.h"
#include "expert_node_v2/expert_format_v2.h"
#include "expert_node_v2/backend/cuda/backend_cuda_v2.h"

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
    const int down_rows = 7168;
    const int down_cols = 2048;
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

    const int down_num_row_blocks = (down_rows + row_block - 1) / row_block;
    const int down_num_col_blocks = (down_cols + col_block - 1) / col_block;

    bundle->w_down.shape = {down_rows, down_cols};
    bundle->w_down.dtype = "torch.float8_e4m3fn";
    bundle->w_down.bytes.resize(static_cast<std::size_t>(down_rows) * down_cols);
    bundle->w_down.ready = true;
    for (std::size_t i = 0; i < bundle->w_down.bytes.size(); ++i) {
        bundle->w_down.bytes[i] = static_cast<std::uint8_t>((i * 13 + 17) & 0xff);
    }

    bundle->w_down_scale.shape = {down_num_row_blocks, down_num_col_blocks};
    bundle->w_down_scale.dtype = "torch.float32";
    bundle->w_down_scale.bytes.resize(
        static_cast<std::size_t>(down_num_row_blocks) *
        static_cast<std::size_t>(down_num_col_blocks) * sizeof(float));
    bundle->w_down_scale.ready = true;
    {
        float* scales = reinterpret_cast<float*>(bundle->w_down_scale.bytes.data());
        for (int rb = 0; rb < down_num_row_blocks; ++rb) {
            for (int cb = 0; cb < down_num_col_blocks; ++cb) {
                scales[rb * down_num_col_blocks + cb] =
                    0.005f + 0.001f * static_cast<float>(rb) +
                    0.0001f * static_cast<float>(cb);
            }
        }
    }
}

template <class T>
static T cast_from_float(float x);

template <>
__half cast_from_float<__half>(float x) {
    return __float2half(x);
}

#if EXPERT_NODE_V2_HAS_CUDA_BF16
template <>
__nv_bfloat16 cast_from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}
#endif

template <class T>
static float cast_to_float(T x);

template <>
float cast_to_float<__half>(__half x) {
    return __half2float(x);
}

#if EXPERT_NODE_V2_HAS_CUDA_BF16
template <>
float cast_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
#endif

static std::vector<float> run_matvec_cpu_reference(
    const HostTensorV2& weight_ht,
    const HostTensorV2& scale_ht,
    int rows,
    int cols,
    const std::vector<float>& x) {
    const int row_block = 128;
    const int col_block = 128;
    const int num_col_blocks = (cols + col_block - 1) / col_block;

    const std::uint8_t* weights = weight_ht.bytes.data();
    const float* scales =
        reinterpret_cast<const float*>(scale_ht.bytes.data());

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
            const float w = decode_torch_e4m3fn_byte(weights[w_idx]) * scales[s_idx];
            sum += w * x[k];
        }
        y[r] = sum;
    }
    return y;
}

static std::vector<float> run_expert_cpu_reference(
    const ExpertTensorBundleV2& bundle,
    const std::vector<float>& x) {
    std::vector<float> up = run_matvec_cpu_reference(bundle.w_up, bundle.w_up_scale, 2048, 7168, x);
    std::vector<float> gate = run_matvec_cpu_reference(bundle.w_gate, bundle.w_gate_scale, 2048, 7168, x);

    std::vector<float> h(2048);
    for (int i = 0; i < 2048; ++i) {
        const float g = gate[i];
        h[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }

    return run_matvec_cpu_reference(bundle.w_down, bundle.w_down_scale, 7168, 2048, h);
}

static float percentile(std::vector<float> v, float p) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    const float idx = p * static_cast<float>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    const float t = idx - static_cast<float>(lo);
    return v[lo] * (1.0f - t) + v[hi] * t;
}

struct Args {
    std::string mode = "correctness";
    std::string dtype = "fp16";
    int warmup = 20;
    int iters = 100;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            a.mode = argv[++i];
        } else if (std::strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            a.dtype = argv[++i];
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            a.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            a.iters = std::atoi(argv[++i]);
        }
    }
    return a;
}

template <class TAct>
int run_main_typed(const Args& args) {
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

    ExpertWorkspaceConfigV2 cfg;
    cfg.hidden_dim = 7168;
    cfg.inter_dim = 2048;

    ExpertWorkspaceCudaV2 ws;
    if (!InitExpertWorkspaceCudaV2(cfg, &ws)) {
        std::printf("InitExpertWorkspaceCudaV2 failed\n");
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    std::vector<float> x_full(cfg.hidden_dim);
    std::vector<TAct> x_dev_host(cfg.hidden_dim);
    std::vector<float> x_quant(cfg.hidden_dim);

    for (int i = 0; i < cfg.hidden_dim; ++i) {
        x_full[i] = std::sin(0.0005f * static_cast<float>(i));
        x_dev_host[i] = cast_from_float<TAct>(x_full[i]);
        x_quant[i] = cast_to_float<TAct>(x_dev_host[i]);
    }

    TAct* d_x = nullptr;
    TAct* d_y = nullptr;

    err = cudaMalloc(reinterpret_cast<void**>(&d_x),
                     static_cast<std::size_t>(cfg.hidden_dim) * sizeof(TAct));
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_x failed: %s\n", cudaGetErrorString(err));
        FreeExpertWorkspaceCudaV2(&ws);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_y),
                     static_cast<std::size_t>(cfg.hidden_dim) * sizeof(TAct));
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_y failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
        FreeExpertWorkspaceCudaV2(&ws);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    err = cudaMemcpy(
        d_x,
        x_dev_host.data(),
        static_cast<std::size_t>(cfg.hidden_dim) * sizeof(TAct),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::printf("cudaMemcpy d_x failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
        cudaFree(d_y);
        FreeExpertWorkspaceCudaV2(&ws);
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    if (args.mode == "correctness") {
        if (!RunExpertCudaV2<TAct>(storage.view(), &ws, d_x, d_y, nullptr)) {
            std::printf("RunExpertCudaV2 failed\n");
            cudaFree(d_x);
            cudaFree(d_y);
            FreeExpertWorkspaceCudaV2(&ws);
            FreeExpertWeightsCudaV2(&storage);
            return 1;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_x);
            cudaFree(d_y);
            FreeExpertWorkspaceCudaV2(&ws);
            FreeExpertWeightsCudaV2(&storage);
            return 1;
        }

        std::vector<TAct> y_gpu_t(cfg.hidden_dim);
        err = cudaMemcpy(
            y_gpu_t.data(),
            d_y,
            static_cast<std::size_t>(cfg.hidden_dim) * sizeof(TAct),
            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::printf("cudaMemcpy y_gpu failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_x);
            cudaFree(d_y);
            FreeExpertWorkspaceCudaV2(&ws);
            FreeExpertWeightsCudaV2(&storage);
            return 1;
        }

        std::vector<float> y_gpu(cfg.hidden_dim);
        for (int i = 0; i < cfg.hidden_dim; ++i) {
            y_gpu[i] = cast_to_float<TAct>(y_gpu_t[i]);
        }

        std::vector<float> y_cpu_full = run_expert_cpu_reference(bundle, x_full);
        std::vector<float> y_cpu_quant = run_expert_cpu_reference(bundle, x_quant);

        float max_abs = 0.0f;
        float sum_abs = 0.0f;
        float dot = 0.0f;
        float norm_cpu = 0.0f;
        float norm_gpu = 0.0f;

        float max_abs_full = 0.0f;
        float sum_abs_full = 0.0f;

        for (int i = 0; i < cfg.hidden_dim; ++i) {
            const float cpu_full_v = y_cpu_full[i];
            const float cpu_q_v = y_cpu_quant[i];
            const float gpu_v = y_gpu[i];

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

        const float mean_abs = sum_abs / static_cast<float>(cfg.hidden_dim);
        const float mean_abs_full = sum_abs_full / static_cast<float>(cfg.hidden_dim);
        const float cos =
            (norm_cpu > 0.0f && norm_gpu > 0.0f)
                ? (dot / (std::sqrt(norm_cpu) * std::sqrt(norm_gpu)))
                : 0.0f;

        std::printf(
            "correctness: q_max_abs=%g q_mean_abs=%g full_max_abs=%g full_mean_abs=%g cos=%g\n",
            max_abs, mean_abs, max_abs_full, mean_abs_full, cos);

        for (int i = 0; i < 8; ++i) {
            std::printf("cpu_full[%d]=%g cpu_q[%d]=%g gpu[%d]=%g\n",
                        i, y_cpu_full[i], i, y_cpu_quant[i], i, y_gpu[i]);
        }
    } else {
        for (int i = 0; i < args.warmup; ++i) {
            if (!RunExpertCudaV2<TAct>(storage.view(), &ws, d_x, d_y, nullptr)) {
                std::printf("warmup failed at iter=%d\n", i);
                cudaFree(d_x);
                cudaFree(d_y);
                FreeExpertWorkspaceCudaV2(&ws);
                FreeExpertWeightsCudaV2(&storage);
                return 1;
            }
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::printf("warmup synchronize failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_x);
            cudaFree(d_y);
            FreeExpertWorkspaceCudaV2(&ws);
            FreeExpertWeightsCudaV2(&storage);
            return 1;
        }

        cudaEvent_t ev_start = nullptr;
        cudaEvent_t ev_stop = nullptr;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);

        std::vector<float> ms_list;
        ms_list.reserve(static_cast<std::size_t>(args.iters));

        for (int i = 0; i < args.iters; ++i) {
            cudaEventRecord(ev_start);
            if (!RunExpertCudaV2<TAct>(storage.view(), &ws, d_x, d_y, nullptr)) {
                std::printf("benchmark failed at iter=%d\n", i);
                cudaEventDestroy(ev_start);
                cudaEventDestroy(ev_stop);
                cudaFree(d_x);
                cudaFree(d_y);
                FreeExpertWorkspaceCudaV2(&ws);
                FreeExpertWeightsCudaV2(&storage);
                return 1;
            }
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            ms_list.push_back(ms);
        }

        float sum = 0.0f;
        for (float v : ms_list) sum += v;
        const float mean = ms_list.empty() ? 0.0f : sum / static_cast<float>(ms_list.size());
        const float p50 = percentile(ms_list, 0.50f);
        const float p90 = percentile(ms_list, 0.90f);
        const float p99 = percentile(ms_list, 0.99f);

        std::printf("CSV_HEADER,kind,dtype,warmup,iters,mean_ms,p50_ms,p90_ms,p99_ms\n");
        std::printf("CSV_ROW,run_expert,%s,%d,%d,%g,%g,%g,%g\n",
                    args.dtype.c_str(), args.warmup, args.iters, mean, p50, p90, p99);

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    FreeExpertWorkspaceCudaV2(&ws);
    FreeExpertWeightsCudaV2(&storage);
    return 0;
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    if (args.dtype == "fp16") {
        return run_main_typed<__half>(args);
    }

#if EXPERT_NODE_V2_HAS_CUDA_BF16
    if (args.dtype == "bf16") {
        return run_main_typed<__nv_bfloat16>(args);
    }
#endif

    std::printf("unsupported dtype: %s\n", args.dtype.c_str());
    return 1;
}
