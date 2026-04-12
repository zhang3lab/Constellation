#include "common/resident_inventory_codec.h"

#include <cstdint>
#include <string>

namespace common {

bool EncodeResidentInventoryReplyBody(
    const std::vector<ResidentInventoryWorkerInfo>& workers,
    std::string* out) {
    if (out == nullptr) return false;
    out->clear();

    std::string body;
    AppendU32(&body, static_cast<std::uint32_t>(workers.size()));

    for (const auto& worker : workers) {
        if (worker.worker_id < 0) {
            return false;
        }

        AppendI32(&body, worker.worker_id);
        AppendU32(&body, static_cast<std::uint32_t>(worker.expert_ids.size()));

        for (std::int32_t expert_id : worker.expert_ids) {
            if (expert_id < 0) {
                return false;
            }
            AppendI32(&body, expert_id);
        }
    }

    *out = std::move(body);
    return true;
}

}  // namespace common
