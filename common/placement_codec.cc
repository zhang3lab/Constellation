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

bool EncodePlacementAck(const PlacementAck& msg, std::string* out) {
    if (out == nullptr) return false;

    out->clear();
    out->reserve(16);

    AppendU32(out, msg.status_code);
    AppendU8(out, msg.needs_reload ? 1 : 0);
    AppendU8(out, msg.all_ready ? 1 : 0);
    AppendU16(out, 0);  // reserved
    AppendU32(out, msg.num_target_experts);
    AppendU32(out, msg.num_ready_experts);
    return true;
}

bool DecodePlacementAck(const void* data, std::size_t len, PlacementAck* out) {
    if (data == nullptr || out == nullptr) return false;
    if (len != 16) return false;

    const auto* p = static_cast<const std::uint8_t*>(data);
    std::size_t off = 0;

    std::uint32_t status_code = 0;
    std::uint8_t needs_reload = 0;
    std::uint8_t all_ready = 0;
    std::uint16_t reserved = 0;
    std::uint32_t num_target_experts = 0;
    std::uint32_t num_ready_experts = 0;

    if (!ReadU32(p, len, &off, &status_code)) return false;
    if (!ReadU8(p, len, &off, &needs_reload)) return false;
    if (!ReadU8(p, len, &off, &all_ready)) return false;
    if (!ReadU16(p, len, &off, &reserved)) return false;
    if (!ReadU32(p, len, &off, &num_target_experts)) return false;
    if (!ReadU32(p, len, &off, &num_ready_experts)) return false;

    out->status_code = status_code;
    out->needs_reload = (needs_reload != 0);
    out->all_ready = (all_ready != 0);
    out->num_target_experts = num_target_experts;
    out->num_ready_experts = num_ready_experts;
    return true;
}

}  // namespace common
