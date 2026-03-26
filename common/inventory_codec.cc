#include "common/inventory_codec.h"

#include <cstdint>
#include <string>

namespace common {

bool EncodeInventoryReplyBody(
    const StaticNodeInfo& static_info,
    const DynamicNodeInfo& dynamic_info,
    std::string* out) {
    if (out == nullptr) return false;
    out->clear();

    if (static_info.node_id != dynamic_info.node_id) {
        return false;
    }

    if (static_info.gpus.size() != dynamic_info.gpus.size()) {
        return false;
    }

    std::string body;
    AppendString(&body, static_info.node_id);
    AppendU32(&body, static_cast<std::uint32_t>(dynamic_info.node_status));
    AppendU32(&body, static_cast<std::uint32_t>(static_info.gpus.size()));

    for (std::size_t i = 0; i < static_info.gpus.size(); ++i) {
        const auto& sgpu = static_info.gpus[i];
        const auto& dgpu = dynamic_info.gpus[i];

        if (sgpu.worker_id != dgpu.worker_id) {
            return false;
        }

        AppendI32(&body, sgpu.worker_id);
        AppendString(&body, sgpu.gpu_name);
        AppendU64(&body, sgpu.total_mem_bytes);
        AppendU64(&body, dgpu.free_mem_bytes);
        AppendU32(&body, sgpu.worker_port);
        AppendU32(&body, static_cast<std::uint32_t>(dgpu.gpu_status));
        AppendU32(&body, static_cast<std::uint32_t>(sgpu.gpu_vendor));
        AppendU32(&body, sgpu.capability_flags);
        AppendString(&body, sgpu.arch_name);
    }

    *out = std::move(body);
    return true;
}

}  // namespace common
