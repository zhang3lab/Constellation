#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/backend/activation_codec_v2.h"
#include "expert_node_v2/backend/cpu/backend_cpu_v2.h"
#include "expert_node_v2/backend/dummy_expert_data_v2.h"
#include "expert_node_v2/backend/expert_reference_v2.h"
#include "expert_node_v2/expert_format_v2.h"

namespace {

struct Args {
    std::string mode = "correctness";
    std::string dtype = "fp16";
    int warmup = 5;
    int iters = 20;
};

Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];

        if (s == "--mode") {
            if (i + 1 >= argc) {
                std::printf("missing value for --mode\n");
                std::exit(1);
            }
            args.mode = argv[++i];
        } else if (s == "--dtype") {
            if (i + 1 >= argc) {
                std::printf("missing value for --dtype\n");
                std::exit(1);
            }
            args.dtype = argv[++i];
        } else if (s == "--warmup") {
            if (i + 1 >= argc) {
                std::printf("missing value for --warmup\n");
                std::exit(1);
            }
            args.warmup = std::atoi(argv[++i]);
        } else if (s == "--iters") {
            if (i + 1 >= argc) {
                std::printf("missing value for --iters\n");
                std::exit(1);
            }
            args.iters = std::atoi(argv[++i]);
        } else {
            std::printf("unknown arg: %s\n", s.c_str());
            std::exit(1);
        }
    }

    if (args.mode != "correctness" &&
        args.mode != "benchmark" &&
        args.mode != "profile") {
        std::printf("unsupported mode: %s\n", args.mode.c_str());
        std::exit(1);
    }

    if (args.dtype != "fp16" && args.dtype != "bf16") {
        std::printf("unsupported dtype: %s\n", args.dtype.c_str());
        std::exit(1);
    }

    if (args.warmup < 0) {
        std::printf("warmup must be >= 0\n");
        std::exit(1);
    }
    if (args.iters <= 0) {
        std::printf("iters must be > 0\n");
        std::exit(1);
    }

    return args;
}

float percentile(std::vector<float> values, float q) {
    if (values.empty()) return 0.0f;
    if (q <= 0.0f) q = 0.0f;
    if (q >= 1.0f) q = 1.0f;

    std::sort(values.begin(), values.end());

    if (values.size() == 1) {
        return values[0];
    }

    const float pos = q * static_cast<float>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));

    if (lo == hi) {
        return values[lo];
    }

    const float w = pos - static_cast<float>(lo);
    return values[lo] * (1.0f - w) + values[hi] * w;
}

struct TestContext {
    common::ActivationDType act_dtype = common::ActivationDType::FP16;

    ExpertTensorBundleV2 bundle;
    ExpertWeightsViewV2 host_view;
    ExpertDeviceStorageV2 storage;
    ExpertWorkspaceConfigV2 cfg;
    ExpertWorkspaceCpuV2 ws;

    std::vector<float> x_float;
    std::vector<std::uint16_t> x_act;
    std::vector<std::uint8_t> y_cpu_bytes;

    bool storage_ready = false;
    bool ws_ready = false;
};

void CleanupTestContext(TestContext* ctx) {
    if (ctx == nullptr) return;

    if (ctx->ws_ready) {
        FreeExpertWorkspaceCpuV2(&ctx->ws);
        ctx->ws_ready = false;
    }
    if (ctx->storage_ready) {
        FreeExpertWeightsCpuV2(&ctx->storage);
        ctx->storage_ready = false;
    }
}

bool InitTestContext(const Args& args, TestContext* ctx) {
    if (ctx == nullptr) return false;

    if (args.dtype == "fp16") {
        ctx->act_dtype = common::ActivationDType::FP16;
    } else if (args.dtype == "bf16") {
        ctx->act_dtype = common::ActivationDType::BF16;
    } else {
        std::printf("unsupported dtype: %s\n", args.dtype.c_str());
        return false;
    }

    FillDummyExpertBundleV2(&ctx->bundle);

    if (!BuildExpertWeightsViewV2(ctx->bundle, &ctx->host_view)) {
        std::printf("BuildExpertWeightsViewV2 failed\n");
        return false;
    }

    if (!UploadExpertCpuV2(0, ctx->bundle, &ctx->storage)) {
        std::printf("UploadExpertCpuV2 failed\n");
        return false;
    }
    ctx->storage_ready = true;

    ctx->cfg.hidden_dim = 7168;
    ctx->cfg.inter_dim = 2048;

    if (!InitExpertWorkspaceCpuV2(ctx->cfg, &ctx->ws)) {
        std::printf("InitExpertWorkspaceCpuV2 failed\n");
        return false;
    }
    ctx->ws_ready = true;

    FillDummyInputActivationV2(
        ctx->cfg.hidden_dim,
        ctx->act_dtype,
        &ctx->x_float,
        &ctx->x_act);

    ctx->y_cpu_bytes.assign(
        static_cast<std::size_t>(ctx->cfg.hidden_dim) * sizeof(std::uint16_t),
        0);

    return true;
}

bool RunCorrectness(const Args& args, TestContext* ctx) {
    (void)args;
    if (ctx == nullptr) return false;

    if (!RunExpertCpuV2(
            ctx->storage.view(),
            &ctx->ws,
            ctx->x_act.data(),
            ctx->act_dtype,
            ctx->y_cpu_bytes.data(),
            ctx->act_dtype)) {
        std::printf("RunExpertCpuV2 failed\n");
        return false;
    }

    std::vector<std::uint8_t> y_ref_bytes;
    std::vector<float> h_ref_debug;
    if (!RunExpertReferenceV2(
            ctx->host_view,
            ctx->x_act.data(),
            ctx->act_dtype,
            ctx->act_dtype,
            &y_ref_bytes,
            &h_ref_debug)) {
        std::printf("RunExpertReferenceV2 failed\n");
        return false;
    }

    const auto* y_ref_u16 =
        reinterpret_cast<const std::uint16_t*>(y_ref_bytes.data());
    const auto* y_cpu_u16 =
        reinterpret_cast<const std::uint16_t*>(ctx->y_cpu_bytes.data());

    std::vector<float> y_ref(ctx->cfg.hidden_dim);
    std::vector<float> y_cpu(ctx->cfg.hidden_dim);
    for (int i = 0; i < ctx->cfg.hidden_dim; ++i) {
        y_ref[i] = DecodeActivationToFloatV2(ctx->act_dtype, y_ref_u16[i]);
        y_cpu[i] = DecodeActivationToFloatV2(ctx->act_dtype, y_cpu_u16[i]);
    }

    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    float dot = 0.0f;
    float norm_ref = 0.0f;
    float norm_cpu = 0.0f;

    for (int i = 0; i < ctx->cfg.hidden_dim; ++i) {
        const float ref_v = y_ref[i];
        const float cpu_v = y_cpu[i];
        const float abs_err = std::fabs(ref_v - cpu_v);

        if (abs_err > max_abs) max_abs = abs_err;
        sum_abs += abs_err;
        dot += ref_v * cpu_v;
        norm_ref += ref_v * ref_v;
        norm_cpu += cpu_v * cpu_v;
    }

    const float mean_abs = sum_abs / static_cast<float>(ctx->cfg.hidden_dim);
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

    return true;
}

void FlushCpuCachesOmpLocalV2() {
    static std::vector<std::uint8_t> buf(512 * 1024 * 1024, 1);
    static volatile std::uint64_t sink = 0;

    std::uint64_t acc = 0;

#pragma omp parallel for reduction(+:acc) schedule(static)
    for (std::size_t i = 0; i < buf.size(); i += 64) {
        buf[i] ^= 1;
        acc += buf[i];
    }

    sink += acc;
}

bool RunBenchmark(const Args& args, TestContext* ctx) {
    if (ctx == nullptr) return false;

    for (int i = 0; i < args.warmup; ++i) {
        if (!RunExpertCpuV2(
                ctx->storage.view(),
                &ctx->ws,
                ctx->x_act.data(),
                ctx->act_dtype,
                ctx->y_cpu_bytes.data(),
                ctx->act_dtype)) {
            std::printf("warmup failed at iter=%d\n", i);
            return false;
        }
    }

    std::vector<float> ms_list;
    ms_list.reserve(static_cast<std::size_t>(args.iters));

    for (int i = 0; i < args.iters; ++i) {
        FlushCpuCachesOmpLocalV2();

        const auto t0 = std::chrono::steady_clock::now();
        if (!RunExpertCpuV2(
                ctx->storage.view(),
                &ctx->ws,
                ctx->x_act.data(),
                ctx->act_dtype,
                ctx->y_cpu_bytes.data(),
                ctx->act_dtype)) {
            std::printf("benchmark failed at iter=%d\n", i);
            return false;
        }
        const auto t1 = std::chrono::steady_clock::now();

        const float ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        ms_list.push_back(ms);
    }

    float sum = 0.0f;
    for (float v : ms_list) sum += v;
    const float mean =
        ms_list.empty() ? 0.0f : sum / static_cast<float>(ms_list.size());
    const float p50 = percentile(ms_list, 0.50f);
    const float p90 = percentile(ms_list, 0.90f);
    const float p99 = percentile(ms_list, 0.99f);

    std::printf("CSV_HEADER,kind,dtype,warmup,iters,mean_ms,p50_ms,p90_ms,p99_ms\n");
    std::printf("CSV_ROW,run_expert_cpu,%s,%d,%d,%g,%g,%g,%g\n",
                args.dtype.c_str(), args.warmup, args.iters,
                mean, p50, p90, p99);

    return true;
}

bool RunProfile(const Args& args, TestContext* ctx) {
    if (ctx == nullptr) return false;

    if (!RunExpertCpuV2(
            ctx->storage.view(),
            &ctx->ws,
            ctx->x_act.data(),
            ctx->act_dtype,
            ctx->y_cpu_bytes.data(),
            ctx->act_dtype)) {
        std::printf("profile warmup RunExpertCpuV2 failed\n");
        return false;
    }

    std::vector<float> run_ms_list;
    std::vector<float> ref_ms_list;
    run_ms_list.reserve(static_cast<std::size_t>(args.iters));
    ref_ms_list.reserve(static_cast<std::size_t>(args.iters));

    std::vector<std::uint8_t> y_ref_bytes;
    std::vector<float> h_ref_debug;

    for (int i = 0; i < args.iters; ++i) {
        {
            const std::clock_t t0 = std::clock();
            if (!RunExpertCpuV2(
                    ctx->storage.view(),
                    &ctx->ws,
                    ctx->x_act.data(),
                    ctx->act_dtype,
                    ctx->y_cpu_bytes.data(),
                    ctx->act_dtype)) {
                std::printf("profile RunExpertCpuV2 failed at iter=%d\n", i);
                return false;
            }
            const std::clock_t t1 = std::clock();
            const float ms =
                1000.0f *
                static_cast<float>(t1 - t0) /
                static_cast<float>(CLOCKS_PER_SEC);
            run_ms_list.push_back(ms);
        }

        {
            const std::clock_t t0 = std::clock();
            if (!RunExpertReferenceV2(
                    ctx->host_view,
                    ctx->x_act.data(),
                    ctx->act_dtype,
                    ctx->act_dtype,
                    &y_ref_bytes,
                    &h_ref_debug)) {
                std::printf("profile RunExpertReferenceV2 failed at iter=%d\n", i);
                return false;
            }
            const std::clock_t t1 = std::clock();
            const float ms =
                1000.0f *
                static_cast<float>(t1 - t0) /
                static_cast<float>(CLOCKS_PER_SEC);
            ref_ms_list.push_back(ms);
        }
    }

    auto print_stats = [](const char* name, const std::vector<float>& ms_list) {
        float sum = 0.0f;
        for (float v : ms_list) sum += v;
        const float mean =
            ms_list.empty() ? 0.0f : sum / static_cast<float>(ms_list.size());
        const float p50 = percentile(ms_list, 0.50f);
        const float p90 = percentile(ms_list, 0.90f);
        const float p99 = percentile(ms_list, 0.99f);

        std::printf(
            "PROFILE,%s,mean_ms=%g,p50_ms=%g,p90_ms=%g,p99_ms=%g\n",
            name,
            mean,
            p50,
            p90,
            p99);
    };

    print_stats("run_expert_cpu", run_ms_list);
    print_stats("run_expert_reference", ref_ms_list);

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    TestContext ctx;
    if (!InitTestContext(args, &ctx)) {
        CleanupTestContext(&ctx);
        return 1;
    }

    bool ok = false;
    if (args.mode == "correctness") {
        ok = RunCorrectness(args, &ctx);
    } else if (args.mode == "benchmark") {
        ok = RunBenchmark(args, &ctx);
    } else if (args.mode == "profile") {
        ok = RunProfile(args, &ctx);
    } else {
        std::printf("unsupported mode: %s\n", args.mode.c_str());
        CleanupTestContext(&ctx);
        return 1;
    }

    CleanupTestContext(&ctx);
    return ok ? 0 : 1;
}
