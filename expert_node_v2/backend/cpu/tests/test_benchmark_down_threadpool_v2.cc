#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <mutex>
#include <functional>
#include <condition_variable>

#include "expert_node_v2/backend/cpu/backend_cpu_v2.h"
#include "expert_node_v2/backend/dummy_expert_data_v2.h"
#include "expert_node_v2/backend/expert_reference_v2.h"
#include "expert_node_v2/backend/fp8_lut_v2.h"
#include "expert_node_v2/expert_format_v2.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// 下面这些类型/函数按你当前工程实际放置位置调整 include。
// 如果它们已经在别的 test 里定义而不是头里，建议直接把定义拷过来。
//
// 需要存在：
// - TestContext
// - InitTestContext(...)
// - CleanupTestContext(...)
// - Fp16ResidentMatrixLocalV2
// - BuildDownFp16ResidentLocalV2(...)
// - FixedRangeThreadPoolV2
// - RunDownCpuFp16ResidentF16cAvx2ThreadPoolV2(...)
// - FlushCpuCachesOmpLocalV2(...)

struct Args {
    std::string config = "server/test/config.json";
    std::string dtype = "";
    int warmup = 5;
    int iters = 100;
    int threads = 4;
    bool flush_cache = false;
};

struct TestContext {
    common::ActivationDType act_dtype = common::ActivationDType::FP16;

    ExpertTensorBundleV2 bundle;
    ExpertWeightsViewV2 host_view;
    ExpertDeviceStorageV2 storage;
    ExpertWorkspaceConfigV2 cfg;
    ExpertWorkspaceCpuV2 ws;

    std::vector<float> x_float;
    std::vector<std::uint16_t> x_act;

    bool storage_ready = false;
    bool ws_ready = false;
};

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
struct Fp16ResidentMatrixLocalV2 {
    int rows = 0;
    int cols = 0;
    std::vector<std::uint16_t> data;  // fp16 payload
};

class FixedRangeThreadPoolV2 {
public:
    using RangeFn = std::function<void(int, int)>;

    FixedRangeThreadPoolV2() = default;
    ~FixedRangeThreadPoolV2() { shutdown(); }

    bool init(int num_threads) {
        if (num_threads <= 0) return false;
        shutdown();

        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = false;
            active_workers_ = 0;
            job_begin_ = 0;
            job_end_ = 0;
            job_epoch_ = 0;
	    worker_last_ms_.assign(static_cast<std::size_t>(num_threads), 0.0);
        }

        try {
            workers_.reserve(static_cast<std::size_t>(num_threads));
            for (int worker_id = 0; worker_id < num_threads; ++worker_id) {
                workers_.emplace_back([this, worker_id]() {
                    worker_loop(worker_id);
                });
            }
        } catch (...) {
            shutdown();
            return false;
        }

        return true;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_job_.notify_all();

        for (auto& th : workers_) {
namespace {

Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string s(argv[i]);

        auto need_value = [&](const char* name) {
            if (i + 1 >= argc) {
                std::printf("missing value for %s\n", name);
                std::exit(1);
            }
        };

        if (s == "--config") {
            need_value("--config");
            args.config = argv[++i];
        } else if (s == "--warmup") {
            need_value("--warmup");
            args.warmup = std::atoi(argv[++i]);
        } else if (s == "--iters") {
            need_value("--iters");
            args.iters = std::atoi(argv[++i]);
        } else if (s == "--threads") {
            need_value("--threads");
            args.threads = std::atoi(argv[++i]);
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

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    TestContext ctx;
    if (!InitTestContext(args, &ctx)) {
        CleanupTestContext(&ctx);
        return 1;
    }

    Fp16ResidentMatrixLocalV2 w_down_fp16;
    if (!BuildDownFp16ResidentLocalV2(ctx.storage.view().w_down, &w_down_fp16)) {
        std::printf("BuildDownFp16ResidentLocalV2 failed\n");
        CleanupTestContext(&ctx);
        return 1;
    }

    FixedRangeThreadPoolV2 pool;
    if (!pool.init(args.threads)) {
        std::printf("FixedRangeThreadPoolV2 init failed\n");
        CleanupTestContext(&ctx);
        return 1;
    }

    std::vector<float> h_mid(static_cast<std::size_t>(ctx.cfg.inter_dim), 0.0f);
    std::vector<float> y_f32(static_cast<std::size_t>(ctx.cfg.hidden_dim), 0.0f);

    // warmup
    for (int i = 0; i < args.warmup; ++i) {
        if (args.flush_cache) {
            FlushCpuCachesOmpLocalV2();
        }

        if (!RunDownCpuFp16ResidentF16cAvx2ThreadPoolV2(
                &pool,
                w_down_fp16,
                h_mid.data(),
                y_f32.data())) {
            std::printf("warmup failed at iter=%d\n", i);
            pool.shutdown();
            CleanupTestContext(&ctx);
            return 1;
        }
    }

    std::vector<float> ms_list;
    ms_list.reserve(static_cast<std::size_t>(args.iters));

    for (int i = 0; i < args.iters; ++i) {
        if (args.flush_cache) {
            FlushCpuCachesOmpLocalV2();
        }

        const auto t0 = std::chrono::steady_clock::now();
        if (!RunDownCpuFp16ResidentF16cAvx2ThreadPoolV2(
                &pool,
                w_down_fp16,
                h_mid.data(),
                y_f32.data())) {
            std::printf("benchmark failed at iter=%d\n", i);
            pool.shutdown();
            CleanupTestContext(&ctx);
            return 1;
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

    std::printf(
        "CSV_HEADER,kind,threads,flush_cache,warmup,iters,mean_ms,p50_ms,p90_ms,p99_ms\n");
    std::printf(
        "CSV_ROW,down_fp16_f16c_avx2_threadpool,%d,%d,%d,%d,%g,%g,%g,%g\n",
        args.threads,
        args.flush_cache ? 1 : 0,
        args.warmup,
        args.iters,
        mean,
        p50,
        p90,
        p99);

    pool.shutdown();
    CleanupTestContext(&ctx);
    return 0;
}
