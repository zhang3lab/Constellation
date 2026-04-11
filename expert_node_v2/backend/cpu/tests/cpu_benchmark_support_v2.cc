#include "expert_node_v2/backend/cpu/tests/cpu_benchmark_support_v2.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <vector>

#include <omp.h>

namespace {

std::vector<int> ParseThreadList(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        const int v = std::atoi(item.c_str());
        if (v > 0) out.push_back(v);
    }

    if (out.empty()) out.push_back(1);
    return out;
}

}  // namespace

Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string s(argv[i]);

        if (s == "--dtype") {
            if (i + 1 >= argc) {
                std::printf("missing value for --dtype\n");
                std::exit(1);
            }
            args.dtype = argv[++i];
        } else if (s == "--iters") {
            if (i + 1 >= argc) {
                std::printf("missing value for --iters\n");
                std::exit(1);
            }
            args.iters = std::atoi(argv[++i]);
        } else if (s == "--config") {
            if (i + 1 >= argc) {
                std::printf("missing value for --config\n");
                std::exit(1);
            }
            args.config = argv[++i];
        } else if (s == "--warmup") {
            if (i + 1 >= argc) {
                std::printf("missing value for --warmup\n");
                std::exit(1);
            }
            args.warmup = std::atoi(argv[++i]);
        } else if (s == "--threads") {
            if (i + 1 >= argc) {
                std::printf("missing value for --threads\n");
                std::exit(1);
            }
            const std::string t = argv[++i];
            args.thread_list = ParseThreadList(t);
            args.threads = args.thread_list.empty() ? 1 : args.thread_list.front();
        } else if (s == "--flush-cache") {
            args.flush_cache = true;
        } else {
            std::printf("unknown arg: %s\n", s.c_str());
            std::exit(1);
        }
    }

    if (args.warmup < 0) args.warmup = 0;
    if (args.iters <= 0) args.iters = 1;
    if (args.threads <= 0) args.threads = 1;

    if (args.thread_list.empty()) {
        args.thread_list.push_back(args.threads);
    }

    return args;
}

float percentile(std::vector<float> xs, float q) {
    if (xs.empty()) return 0.0f;

    q = std::max(0.0f, std::min(1.0f, q));
    std::sort(xs.begin(), xs.end());

    const float pos = q * static_cast<float>(xs.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(pos);
    const std::size_t hi = std::min(lo + 1, xs.size() - 1);
    const float t = pos - static_cast<float>(lo);

    return xs[lo] * (1.0f - t) + xs[hi] * t;
}

void print_stats(const char* name, const std::vector<float>& xs) {
    if (xs.empty()) {
        std::printf("PROFILE,%s,mean_ms=0,p50_ms=0,p90_ms=0,p99_ms=0\n", name);
        return;
    }

    float sum = 0.0f;
    for (float v : xs) sum += v;

    const float mean = sum / static_cast<float>(xs.size());
    const float p50 = percentile(xs, 0.50f);
    const float p90 = percentile(xs, 0.90f);
    const float p99 = percentile(xs, 0.99f);

    std::printf("PROFILE,%s,mean_ms=%g,p50_ms=%g,p90_ms=%g,p99_ms=%g\n",
                name, mean, p50, p90, p99);
}

void print_stats_with_threads(
    const char* name,
    int threads,
    const std::vector<float>& xs) {
    if (xs.empty()) {
        std::printf(
            "PROFILE,%s,threads=%d,mean_ms=0,p50_ms=0,p90_ms=0,p99_ms=0\n",
            name, threads);
        return;
    }

    float sum = 0.0f;
    for (float v : xs) sum += v;

    const float mean = sum / static_cast<float>(xs.size());
    const float p50 = percentile(xs, 0.50f);
    const float p90 = percentile(xs, 0.90f);
    const float p99 = percentile(xs, 0.99f);

    std::printf(
        "PROFILE,%s,threads=%d,mean_ms=%g,p50_ms=%g,p90_ms=%g,p99_ms=%g\n",
        name, threads, mean, p50, p90, p99);
}

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

void FlushCpuCachesOmpLocalV2() {
    constexpr std::size_t kFlushBytes = 256ull * 1024ull * 1024ull;
    constexpr std::size_t kElems = kFlushBytes / sizeof(std::uint64_t);

    static std::vector<std::uint64_t> buf(kElems, 0);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < buf.size(); ++i) {
        buf[i] += 1;
    }
}
