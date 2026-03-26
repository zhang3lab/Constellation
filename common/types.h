#pragma once

#include <array>
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

enum class GpuVendor : std::uint32_t {
    Unknown = 0,
    Nvidia = 1,
    AMD = 2,
    Intel = 3,
};

constexpr std::uint32_t kGpuCapFp16 = 1u << 0;
constexpr std::uint32_t kGpuCapBf16 = 1u << 1;
constexpr std::uint32_t kGpuCapFp8  = 1u << 2;

struct VendorWorkerSpan {
    std::int32_t worker_id_begin = -1;
    std::int32_t worker_count = 0;
};

struct StaticGpuInfo {
    std::int32_t worker_id = -1;

    std::string gpu_name;
    std::uint64_t total_mem_bytes = 0;
    std::uint32_t worker_port = 0;

    GpuVendor gpu_vendor = GpuVendor::Unknown;
    std::uint32_t capability_flags = 0;
    std::string arch_name;
};

struct StaticNodeInfo {
    std::string node_id;
    std::string host;
    std::int32_t control_port = -1;

    std::array<VendorWorkerSpan, 256> vendor_spans{};
    std::vector<StaticGpuInfo> gpus;
};

struct DynamicGpuInfo {
    std::int32_t worker_id = -1;
    std::uint64_t free_mem_bytes = 0;
    GpuStatus gpu_status = GpuStatus::Idle;
};

struct DynamicNodeInfo {
    std::string node_id;
    NodeStatus node_status = NodeStatus::Booting;
    std::vector<DynamicGpuInfo> gpus;
};
