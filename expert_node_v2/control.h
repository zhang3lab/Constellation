#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common/types.h"
#include "common/protocol.h"
#include "expert_node_v2/expert_registry_v2.h"

struct ActiveLoad {
    bool active = false;
    int expert_id = -1;
    int worker_id = -1;
    common::TensorKind tensor_kind = common::TensorKind::WUp;
    std::uint64_t total_bytes = 0;
    std::uint64_t received_bytes = 0;
    std::vector<std::uint8_t> buffer;
    common::TensorMeta meta;
};

struct PendingResidentBuild {
    int expert_id = -1;
    int worker_id = -1;
    common::GpuVendor vendor = common::GpuVendor::Unknown;
};

struct ResidentBuildState {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<PendingResidentBuild> pending_builds;
    std::unordered_set<std::uint64_t> pending_build_keys;
    bool stop_worker = false;
    std::thread worker;
};

struct ControlState {
    common::NodeStatus node_status = common::NodeStatus::Booting;
    common::StaticNodeInfo static_info;
    ActiveLoad active_load;
    expert_node_v2::ExpertRegistryV2 registry;
    std::shared_mutex mu;
    ResidentBuildState resident_build;
};

void RunControlLoop(ControlState* state);
