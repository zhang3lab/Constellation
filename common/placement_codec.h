#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace common {

struct PlacementAssignment {
    std::int32_t expert_id = -1;
    std::int32_t local_gpu_id = -1;
};

std::string EncodePlacementPlanBody(
    const std::vector<PlacementAssignment>& assignments);

bool DecodePlacementPlanBody(
    const std::string& body,
    std::vector<PlacementAssignment>* out);

}  // namespace common
