#include "common/inventory_codec.h"

#include <cstdint>
#include <string>

namespace common {

std::string EncodeInventoryReplyBody(
    const NodeInfo& node,
    NodeStatus node_status) {
    std::string body;
    body.reserve(256 + node.gpus.size() * 128);

    AppendBytes(&body, node.node_id);
    AppendU32(&body, static_cast<std::uint32_t>(node_status));
    AppendU32(&body, static_cast<std::uint32_t>(node.gpus.size()));

    for (const auto& gpu : node.gpus) {
        AppendI32(&body, gpu.local_gpu_id);

        AppendBytes(&body, gpu.gpu_name);

        AppendU64(&body, gpu.total_mem_bytes);
        AppendU64(&body, gpu.free_mem_bytes);
        AppendU32(&body, gpu.worker_port);
        AppendU32(&body, static_cast<std::uint32_t>(gpu.gpu_status));

        AppendU32(&body, static_cast<std::uint32_t>(gpu.gpu_vendor));
        AppendU32(&body, gpu.capability_flags);

        AppendBytes(&body, gpu.arch_name);
    }

    return body;
}

}  // namespace common
