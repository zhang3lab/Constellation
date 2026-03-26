#include "common/placement_codec.h"

#include <cstddef>
#include <cstdint>

#include "common/protocol.h"

namespace common {

std::string EncodePlacementPlanBody(
    const std::vector<PlacementAssignment>& assignments) {
    std::string body;
    body.reserve(4 + assignments.size() * 8);

    AppendU32(&body, static_cast<std::uint32_t>(assignments.size()));
    for (const auto& a : assignments) {
        AppendI32(&body, a.expert_id);
        AppendI32(&body, a.worker_id);
    }
    return body;
}

bool DecodePlacementPlanBody(
    const std::string& body,
    std::vector<PlacementAssignment>* out) {
    if (out == nullptr) return false;
    out->clear();

    std::size_t offset = 0;
    std::uint32_t n = 0;
    if (!ReadU32(body, &offset, &n)) return false;

    out->reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        PlacementAssignment a;
        if (!ReadI32(body, &offset, &a.expert_id)) return false;
        if (!ReadI32(body, &offset, &a.worker_id)) return false;
        out->push_back(a);
    }

    return offset == body.size();
}

}  // namespace common
