#pragma once

#include <cstdint>
#include <string>

#include "common/protocol.h"

namespace common {

struct LoadWeightsBeginMsg {
    std::int32_t expert_id = -1;
    std::int32_t worker_id = -1;
    TensorKind tensor_kind = TensorKind::WUp;
    std::uint64_t total_bytes = 0;
};

struct LoadWeightsChunkMsg {
    std::int32_t expert_id = -1;
    std::int32_t worker_id = -1;
    TensorKind tensor_kind = TensorKind::WUp;
    std::uint64_t chunk_offset = 0;
    std::vector<std::uint8_t> chunk_data;
};

struct LoadWeightsEndMsg {
    std::int32_t expert_id = -1;
    std::int32_t worker_id = -1;
    TensorKind tensor_kind = TensorKind::WUp;
};

std::string EncodeLoadWeightsBeginBody(const LoadWeightsBeginMsg& msg);

bool DecodeLoadWeightsBeginBody(
    const std::string& body,
    LoadWeightsBeginMsg* out);

std::string EncodeLoadWeightsChunkBody(const LoadWeightsChunkMsg& msg);

bool DecodeLoadWeightsChunkBody(
    const std::string& body,
    LoadWeightsChunkMsg* out);

std::string EncodeLoadWeightsEndBody(const LoadWeightsEndMsg& msg);

bool DecodeLoadWeightsEndBody(
    const std::string& body,
    LoadWeightsEndMsg* out);

}  // namespace common
