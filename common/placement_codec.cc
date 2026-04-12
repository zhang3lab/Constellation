#include "common/placement_codec.h"

#include <cstddef>
#include <cstdint>

#include "common/protocol.h"

namespace common {

std::string EncodePlacementPlanBody(const PlacementPlan& plan) {
    std::string body;
    body.reserve(8 + plan.assignments.size() * 8);

    AppendU32(&body, plan.drop_non_target_residents ? 1u : 0u);
    AppendU32(&body, static_cast<std::uint32_t>(plan.assignments.size()));

    for (const auto& a : plan.assignments) {
        AppendI32(&body, a.expert_id);
        AppendI32(&body, a.worker_id);
    }
    return body;
}

bool DecodePlacementPlanBody(
    const std::string& body,
    PlacementPlan* out) {
    if (out == nullptr) return false;
    out->drop_non_target_residents = false;
    out->assignments.clear();

    std::size_t offset = 0;
    std::uint32_t drop_non_target_residents = 0;
    std::uint32_t n = 0;

    if (!ReadU32(body, &offset, &drop_non_target_residents)) return false;
    if (!ReadU32(body, &offset, &n)) return false;

    out->drop_non_target_residents = (drop_non_target_residents != 0);

    out->assignments.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        PlacementAssignment a;
        if (!ReadI32(body, &offset, &a.expert_id)) return false;
        if (!ReadI32(body, &offset, &a.worker_id)) return false;
        out->assignments.push_back(a);
    }

    return offset == body.size();
}

bool EncodePlacementAck(const PlacementAck& msg, std::string* out) {
    if (out == nullptr) return false;

    out->clear();
    out->reserve(16);

    AppendU32(out, msg.status_code);
    AppendU16(out, msg.needs_load ? 1 : 0);
    AppendU16(out, msg.all_ready ? 1 : 0);
    AppendU32(out, msg.num_target_experts);
    AppendU32(out, msg.num_ready_experts);
    return true;
}

bool DecodePlacementAck(const void* data, std::size_t len, PlacementAck* out) {
    if (data == nullptr || out == nullptr) return false;
    if (len != 16) return false;

    const std::string buf(static_cast<const char*>(data), len);
    std::size_t off = 0;

    std::uint32_t status_code = 0;
    std::uint16_t needs_load = 0;
    std::uint16_t all_ready = 0;
    std::uint32_t num_target_experts = 0;
    std::uint32_t num_ready_experts = 0;

    if (!ReadU32(buf, &off, &status_code)) return false;
    if (!ReadU16(buf, &off, &needs_load)) return false;
    if (!ReadU16(buf, &off, &all_ready)) return false;
    if (!ReadU32(buf, &off, &num_target_experts)) return false;
    if (!ReadU32(buf, &off, &num_ready_experts)) return false;

    out->status_code = status_code;
    out->needs_load = (needs_load != 0);
    out->all_ready = (all_ready != 0);
    out->num_target_experts = num_target_experts;
    out->num_ready_experts = num_ready_experts;
    return true;
}

}  // namespace common
