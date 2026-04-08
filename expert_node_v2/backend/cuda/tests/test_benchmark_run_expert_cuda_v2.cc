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
#include <type_traits>
#include <vector>

#include "expert_node_v2/build_config_v2.h"
#include "expert_node_v2/backend/expert_reference_v2.h"
#include "expert_node_v2/backend/dummy_expert_data_v2.h"
#include "expert_node_v2/expert_format_v2.h"
#include "expert_node_v2/backend/cuda/backend_cuda_v2.h"

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

    ExpertWorkspaceConfigV2 cfg;
    cfg.hidden_dim = 7168;
    cfg.inter_dim = 2048;

    ExpertWorkspaceCudaV2 ws;
    if (!InitExpertWorkspaceCudaV2(cfg, &ws)) {
        std::printf("InitExpertWorkspaceCudaV2 failed\n");
        FreeExpertWeightsCudaV2(&storage);
        return 1;
    }

    std::vector<float> x_float;
    std::vector<std::uint16_t> x_act_u16;
    const common::ActivationDType act_dtype =
        std::is_same_v<TAct, __half>
            ? common::ActivationDType::FP16
#if EXPERT_NODE_V2_HAS_CUDA_BF16
            : common::ActivationDType::BF16;
#else
            : common::ActivationDType::FP16;
#endif

    FillDummyInputActivationV2(
        cfg.hidden_dim,
        act_dtype,
        &x_float,
        &x_act_u16);

    std::vector<TAct> x_dev_host(cfg.hidden_dim);
    for (int i = 0; i < cfg.hidden_dim; ++i) {
        x_dev_host[i] = cast_from_float<TAct>(x_float[i]);
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
        if (!RunExpertCudaV2<TAct, TAct>(storage.view(), &ws, d_x, d_y, nullptr)) {
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

        std::vector<std::uint8_t> y_ref_bytes;
        std::vector<float> h_ref_debug;
        if (!RunExpertReferenceV2(
                host_view,
                x_act_u16.data(),
                act_dtype,
                act_dtype,
                &y_ref_bytes,
                &h_ref_debug)) {
            std::printf("RunExpertReferenceV2 failed\n");
            cudaFree(d_x);
            cudaFree(d_y);
            FreeExpertWorkspaceCudaV2(&ws);
            FreeExpertWeightsCudaV2(&storage);
            return 1;
        }

        const auto* y_ref_t =
            reinterpret_cast<const TAct*>(y_ref_bytes.data());

        std::vector<float> y_ref(cfg.hidden_dim);
        for (int i = 0; i < cfg.hidden_dim; ++i) {
            y_ref[i] = cast_to_float<TAct>(y_ref_t[i]);
        }

        float max_abs = 0.0f;
        float sum_abs = 0.0f;
        float dot = 0.0f;
        float norm_ref = 0.0f;
        float norm_gpu = 0.0f;

        for (int i = 0; i < cfg.hidden_dim; ++i) {
            const float ref_v = y_ref[i];
            const float gpu_v = y_gpu[i];

            const float abs_err = std::fabs(ref_v - gpu_v);
            if (abs_err > max_abs) max_abs = abs_err;

            sum_abs += abs_err;
            dot += ref_v * gpu_v;
            norm_ref += ref_v * ref_v;
            norm_gpu += gpu_v * gpu_v;
        }

        const float mean_abs = sum_abs / static_cast<float>(cfg.hidden_dim);
        const float cos =
            (norm_ref > 0.0f && norm_gpu > 0.0f)
                ? (dot / (std::sqrt(norm_ref) * std::sqrt(norm_gpu)))
                : 0.0f;

        std::printf(
            "correctness: max_abs=%g mean_abs=%g cos=%g\n",
            max_abs, mean_abs, cos);

        for (int i = 0; i < 8; ++i) {
            std::printf("ref[%d]=%g gpu[%d]=%g\n",
                        i, y_ref[i], i, y_gpu[i]);
        }
    } else {
        for (int i = 0; i < args.warmup; ++i) {
            if (!RunExpertCudaV2<TAct, TAct>(storage.view(), &ws, d_x, d_y, nullptr)) {
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
            if (!RunExpertCudaV2<TAct, TAct>(storage.view(), &ws, d_x, d_y, nullptr)) {
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
