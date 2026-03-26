#pragma once

#include <string>

#include "common/types.h"

bool BuildStaticNodeInfo(
    const std::string& node_id,
    const std::string& host,
    int control_port,
    int worker_port_base,
    common::StaticNodeInfo* out);

bool BuildDynamicNodeInfo(
    const common::StaticNodeInfo& static_info,
    common::NodeStatus node_status,
    common::DynamicNodeInfo* out);
