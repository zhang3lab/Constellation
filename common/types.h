#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace common {

// These are in-memory shared concept/types for C++ code.
// They are NOT wire-format structs and must not be sent via raw memcpy.

enum class NodeStatus : std::uint32_t {
    Booting = 0,
    Registered = 1,
    Allocated = 2,
    Loading = 3,
    Ready = 4,
    Serving = 5,
    Failed = 6,
};

enum class GpuStatus : std::uint32_t {
    Init = 0,
    Idle = 1,
    Loading = 2,
    Ready = 3,
    Busy = 4,
    Failed = 5,
};

enum class ExpertStatus : std::uint32_t {
    Empty = 0,
    Assigned = 1,
    Packing = 2,
    Uploading = 3,
    Ready = 4,
    Failed = 5,
};

struct GpuInfo {
    std::int32_t local_gpu_id = -1;
    std::string gpu_name;

    std::uint64_t total_mem_bytes = 0;
    std::uint64_t free_mem_bytes = 0;
    std::uint32_t worker_port = 0;
    GpuStatus gpu_status = GpuStatus::Unknown;

    GpuVendor gpu_vendor = GpuVendor::Unknown;
    std::uint32_t capability_flags = 0;
    std::string arch_name;
};

struct NodeInfo {
    std::string node_id;
    std::string host;
    std::int32_t control_port = -1;
    std::vector<GpuInfo> gpus;
};

struct ExpertPlacement {
    std::int32_t expert_id = -1;
    std::string node_id;
    std::string gpu_uid;
    std::int32_t local_gpu_id = -1;
};

}  // namespace common
