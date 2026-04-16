#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "common/protocol.h"

namespace common {

struct LoadWeightsBeginMsg {
    std::int32_t expert_id = -1;
    std::int32_t worker_id = -1;
    TensorKind tensor_kind = TensorKind::WUp;
    std::uint64_t total_bytes = 0;
    TensorMeta meta;
};

struct LoadWeightsChunkMsgHeader {
    std::int32_t expert_id = -1;
    std::int32_t worker_id = -1;
    TensorKind tensor_kind = TensorKind::WUp;
    std::uint64_t chunk_offset = 0;
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

std::string EncodeLoadWeightsChunkBody(
    const LoadWeightsChunkMsgHeader& header,
    std::span<const std::uint8_t> chunk_view);

bool DecodeLoadWeightsChunkBody(
    const std::string& body,
    LoadWeightsChunkMsgHeader* out_header,
    std::span<const std::uint8_t>* out_chunk_view);

std::string EncodeLoadWeightsEndBody(const LoadWeightsEndMsg& msg);

bool DecodeLoadWeightsEndBody(
    const std::string& body,
    LoadWeightsEndMsg* out);

}  // namespace common
