#pragma once

#include <cstdint>
#include <string>
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
};

struct ControlState {
    common::NodeStatus node_status = common::NodeStatus::Booting;

    common::StaticNodeInfo static_info;

    ActiveLoad active_load;
    expert_node_v2::ExpertRegistryV2 registry;
};

void RunControlLoop(ControlState* state);
