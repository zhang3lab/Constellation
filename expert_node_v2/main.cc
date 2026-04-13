#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "expert_node_v2/control.h"
#include "expert_node_v2/node_info.h"
#include "expert_node_v2/worker.h"

namespace {

struct MainOptions {
    std::string node_id = "node0";
    std::string host = "127.0.0.1";
    int control_port = 40000;
    int worker_base_port = 50000;
    bool verbose = false;
};

bool ParseIntArg(const char* s, int* out) {
    if (s == nullptr || out == nullptr || *s == '\0') return false;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    if (v <= 0 || v > 65535) return false;
    *out = static_cast<int>(v);
    return true;
}

void PrintUsage(const char* prog) {
    std::fprintf(
        stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --node-id <str>           Node id (default: node0)\n"
        "  --host <str>              Advertised host (default: 127.0.0.1)\n"
        "  --control-port <int>      Control port (default: 40000)\n"
        "  --worker-base-port <int>  Worker base port (default: 50000)\n"
        "  --verbose                 Enable verbose control/protocol logs (default: off)\n"
        "  --help                    Show this message\n",
        prog);
}

bool ParseMainOptions(int argc, char** argv, MainOptions* opt) {
    if (opt == nullptr) return false;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--help") == 0) {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (std::strcmp(a, "--node-id") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for --node-id\n");
                return false;
            }
            opt->node_id = argv[++i];
        } else if (std::strcmp(a, "--host") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for --host\n");
                return false;
            }
            opt->host = argv[++i];
        } else if (std::strcmp(a, "--control-port") == 0) {
            if (i + 1 >= argc || !ParseIntArg(argv[i + 1], &opt->control_port)) {
                std::fprintf(stderr, "invalid value for --control-port\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(a, "--worker-base-port") == 0) {
            if (i + 1 >= argc || !ParseIntArg(argv[i + 1], &opt->worker_base_port)) {
                std::fprintf(stderr, "invalid value for --worker-base-port\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(a, "--verbose") == 0) {
            opt->verbose = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a);
            return false;
        }
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    MainOptions opt;
    if (!ParseMainOptions(argc, argv, &opt)) {
        PrintUsage(argv[0]);
        return 1;
    }

    ControlState state;
    state.verbose = opt.verbose;

    if (!BuildStaticNodeInfo(
            opt.node_id,
            opt.host,
            opt.control_port,
            opt.worker_base_port,
            &state.static_info)) {
        std::fprintf(stderr, "BuildStaticNodeInfo failed\n");
        return 1;
    }

    std::printf("[main] node_id=%s host=%s control_port=%d worker_base_port=%d num_gpus=%zu verbose=%d\n",
                state.static_info.node_id.c_str(),
                opt.host.c_str(),
                state.static_info.control_port,
                opt.worker_base_port,
                state.static_info.gpus.size(),
                static_cast<int>(state.verbose));

    std::thread control_thread([&state]() {
        RunControlLoop(&state);
    });

    std::vector<std::unique_ptr<GpuWorkerContextV2>> worker_contexts;
    std::vector<std::thread> worker_threads;

    worker_contexts.reserve(state.static_info.gpus.size());
    worker_threads.reserve(state.static_info.gpus.size());

    for (const auto& gpu : state.static_info.gpus) {
        auto ctx = std::make_unique<GpuWorkerContextV2>();
        ctx->worker_id = gpu.worker_id;
        ctx->worker_port = static_cast<int>(gpu.worker_port);
        ctx->state = &state;

        std::printf("[main] starting gpu worker worker_id=%d port=%d name=%s vendor=%u\n",
                    ctx->worker_id,
                    ctx->worker_port,
                    gpu.gpu_name.c_str(),
                    static_cast<unsigned>(gpu.gpu_vendor));

        worker_threads.emplace_back([raw = ctx.get()]() {
            RunGpuWorkerLoopV2(raw);
        });

        worker_contexts.push_back(std::move(ctx));
    }

    control_thread.join();
    for (auto& t : worker_threads) {
        t.join();
    }

    return 0;
}
