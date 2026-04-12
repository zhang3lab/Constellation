#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/types.h"

namespace common {

std::string EncodePlacementPlanBody(const PlacementPlan& plan);

bool DecodePlacementPlanBody(
    const std::string& body,
    PlacementPlan* out);

bool EncodePlacementAck(const PlacementAck& msg, std::string* out);

bool DecodePlacementAck(const void* data, std::size_t len, PlacementAck* out);

}  // namespace common
