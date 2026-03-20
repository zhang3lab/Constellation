#pragma once

#include <cstdint>
#include <string>

#include "common/protocol.h"

namespace common {

struct LoadWeightsBeginMsg {
    std::int32_t expert_id = -1;
    std::int32_t local_gpu_id = -1;
    TensorKind tensor_kind = TensorKind::WUp;
    std::uint64_t total_bytes = 0;
};

std::string EncodeLoadWeightsBeginBody(const LoadWeightsBeginMsg& msg);

bool DecodeLoadWeightsBeginBody(
    const std::string& body,
    LoadWeightsBeginMsg* out);

}  // namespace common
