#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/backend/cpu/backend_cpu_v2.h"
#include "expert_node_v2/backend/dummy_expert_data_v2.h"
#include "expert_node_v2/expert_format_v2.h"

struct Args {
    std::string config = "server/test/config.json";
    std::string dtype = "fp16";

    int warmup = 5;
    int iters = 100;

    int threads = 1;
    std::vector<int> thread_list;

    bool flush_cache = false;
};

Args parse_args(int argc, char** argv);

float percentile(std::vector<float> xs, float q);

void print_stats(const char* name, const std::vector<float>& xs);

void print_stats_with_threads(
    const char* name,
    int threads,
    const std::vector<float>& xs);

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

bool InitTestContext(const Args& args, TestContext* ctx);

void CleanupTestContext(TestContext* ctx);

void FlushCpuCachesOmpLocalV2();
