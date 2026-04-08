#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/backend/activation_codec_v2.h"
#include "expert_node_v2/backend/cpu/backend_cpu_v2.h"
#include "expert_node_v2/backend/dummy_expert_data_v2.h"
#include "expert_node_v2/backend/expert_reference_v2.h"
#include "expert_node_v2/expert_format_v2.h"

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
    int warmup = 5;
    int iters = 20;
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

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    common::ActivationDType act_dtype = common::ActivationDType::FP16;
    if (args.dtype == "fp16") {
        act_dtype = common::ActivationDType::FP16;
    } else if (args.dtype == "bf16") {
        act_dtype = common::ActivationDType::BF16;
    } else {
        std::printf("unsupported dtype: %s\n", args.dtype.c_str());
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
    if (!UploadExpertCpuV2(0, bundle, &storage)) {
        std::printf("UploadExpertCpuV2 failed\n");
        return 1;
    }

    ExpertWorkspaceConfigV2 cfg;
    cfg.hidden_dim = 7168;
    cfg.inter_dim = 2048;

    ExpertWorkspaceCpuV2 ws;
    if (!InitExpertWorkspaceCpuV2(cfg, &ws)) {
        std::printf("InitExpertWorkspaceCpuV2 failed\n");
        FreeExpertWeightsCpuV2(&storage);
        return 1;
    }

    std::vector<float> x_float;
    std::vector<std::uint16_t> x_act;
    FillDummyInputActivationV2(
        cfg.hidden_dim,
        act_dtype,
        &x_float,
        &x_act);

    std::vector<std::uint8_t> y_cpu_bytes(
        static_cast<std::size_t>(cfg.hidden_dim) * sizeof(std::uint16_t), 0);

    if (args.mode == "correctness") {
        if (!RunExpertCpuV2(
                storage.view(),
                &ws,
                x_act.data(),
                act_dtype,
                y_cpu_bytes.data(),
                act_dtype)) {
            std::printf("RunExpertCpuV2 failed\n");
            FreeExpertWorkspaceCpuV2(&ws);
            FreeExpertWeightsCpuV2(&storage);
            return 1;
        }

        std::vector<std::uint8_t> y_ref_bytes;
        std::vector<float> h_ref_debug;
        if (!RunExpertReferenceV2(
                host_view,
                x_act.data(),
                act_dtype,
                act_dtype,
                &y_ref_bytes,
                &h_ref_debug)) {
            std::printf("RunExpertReferenceV2 failed\n");
            FreeExpertWorkspaceCpuV2(&ws);
            FreeExpertWeightsCpuV2(&storage);
            return 1;
        }

        const auto* y_ref_u16 =
            reinterpret_cast<const std::uint16_t*>(y_ref_bytes.data());
        const auto* y_cpu_u16 =
            reinterpret_cast<const std::uint16_t*>(y_cpu_bytes.data());

        std::vector<float> y_ref(cfg.hidden_dim);
        std::vector<float> y_cpu(cfg.hidden_dim);
        for (int i = 0; i < cfg.hidden_dim; ++i) {
            y_ref[i] = DecodeActivationToFloatV2(act_dtype, y_ref_u16[i]);
            y_cpu[i] = DecodeActivationToFloatV2(act_dtype, y_cpu_u16[i]);
        }

        float max_abs = 0.0f;
        float sum_abs = 0.0f;
        float dot = 0.0f;
        float norm_ref = 0.0f;
        float norm_cpu = 0.0f;

        for (int i = 0; i < cfg.hidden_dim; ++i) {
            const float ref_v = y_ref[i];
            const float cpu_v = y_cpu[i];
            const float abs_err = std::fabs(ref_v - cpu_v);

            if (abs_err > max_abs) max_abs = abs_err;
            sum_abs += abs_err;
            dot += ref_v * cpu_v;
            norm_ref += ref_v * ref_v;
            norm_cpu += cpu_v * cpu_v;
        }

        const float mean_abs = sum_abs / static_cast<float>(cfg.hidden_dim);
        const float cos =
            (norm_ref > 0.0f && norm_cpu > 0.0f)
                ? (dot / (std::sqrt(norm_ref) * std::sqrt(norm_cpu)))
                : 0.0f;

        std::printf(
            "correctness: max_abs=%g mean_abs=%g cos=%g\n",
            max_abs, mean_abs, cos);

        for (int i = 0; i < 8; ++i) {
            std::printf("ref[%d]=%g cpu[%d]=%g\n",
                        i, y_ref[i], i, y_cpu[i]);
        }
    } else {
        for (int i = 0; i < args.warmup; ++i) {
            if (!RunExpertCpuV2(
                    storage.view(),
                    &ws,
                    x_act.data(),
                    act_dtype,
                    y_cpu_bytes.data(),
                    act_dtype)) {
                std::printf("warmup failed at iter=%d\n", i);
                FreeExpertWorkspaceCpuV2(&ws);
                FreeExpertWeightsCpuV2(&storage);
                return 1;
            }
        }

        std::vector<float> ms_list;
        ms_list.reserve(static_cast<std::size_t>(args.iters));

        for (int i = 0; i < args.iters; ++i) {
            const std::clock_t t0 = std::clock();
            if (!RunExpertCpuV2(
                    storage.view(),
                    &ws,
                    x_act.data(),
                    act_dtype,
                    y_cpu_bytes.data(),
                    act_dtype)) {
                std::printf("benchmark failed at iter=%d\n", i);
                FreeExpertWorkspaceCpuV2(&ws);
                FreeExpertWeightsCpuV2(&storage);
                return 1;
            }
            const std::clock_t t1 = std::clock();
            const float ms =
                1000.0f *
                static_cast<float>(t1 - t0) /
                static_cast<float>(CLOCKS_PER_SEC);
            ms_list.push_back(ms);
        }

        float sum = 0.0f;
        for (float v : ms_list) sum += v;
        const float mean = ms_list.empty() ? 0.0f : sum / static_cast<float>(ms_list.size());
        const float p50 = percentile(ms_list, 0.50f);
        const float p90 = percentile(ms_list, 0.90f);
        const float p99 = percentile(ms_list, 0.99f);

        std::printf("CSV_HEADER,kind,dtype,warmup,iters,mean_ms,p50_ms,p90_ms,p99_ms\n");
        std::printf("CSV_ROW,run_expert_cpu,%s,%d,%d,%g,%g,%g,%g\n",
                    args.dtype.c_str(), args.warmup, args.iters,
                    mean, p50, p90, p99);
    }

    FreeExpertWorkspaceCpuV2(&ws);
    FreeExpertWeightsCpuV2(&storage);
    return 0;
}
