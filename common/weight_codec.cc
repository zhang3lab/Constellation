#include "common/weight_codec.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "common/protocol.h"

namespace common {

std::string EncodeLoadWeightsBeginBody(const LoadWeightsBeginMsg& msg) {
    std::string body;
    body.reserve(
        4 + 4 + 4 + 8 + 4 +
        msg.meta.shape.size() * 8 +
        msg.meta.dtype.size() + 4 + 4 + 8);

    AppendI32(&body, msg.expert_id);
    AppendI32(&body, msg.worker_id);
    AppendI32(&body, static_cast<std::int32_t>(msg.tensor_kind));
    AppendU64(&body, msg.total_bytes);

    AppendU32(&body, static_cast<std::uint32_t>(msg.meta.shape.size()));
    for (std::uint64_t d : msg.meta.shape) {
        AppendU64(&body, d);
    }

    AppendString(&body, msg.meta.dtype);
    AppendU32(&body, msg.meta.row_block);
    AppendU32(&body, msg.meta.col_block);

    return body;
}

bool DecodeLoadWeightsBeginBody(
    const std::string& body,
    LoadWeightsBeginMsg* out) {
    if (out == nullptr) return false;

    std::size_t offset = 0;
    if (!ReadI32(body, &offset, &out->expert_id)) return false;
    if (!ReadI32(body, &offset, &out->worker_id)) return false;

    std::int32_t tensor_kind_raw = -1;
    if (!ReadI32(body, &offset, &tensor_kind_raw)) return false;
    if (!ReadU64(body, &offset, &out->total_bytes)) return false;

    if (tensor_kind_raw < static_cast<std::int32_t>(TensorKind::WUp) ||
        tensor_kind_raw > static_cast<std::int32_t>(TensorKind::WDownScale)) {
        return false;
    }
    out->tensor_kind = static_cast<TensorKind>(tensor_kind_raw);

    std::uint32_t ndim = 0;
    if (!ReadU32(body, &offset, &ndim)) return false;
    if (ndim > 16) return false;

    out->meta.shape.clear();
    out->meta.shape.resize(ndim);
    for (std::uint32_t i = 0; i < ndim; ++i) {
        if (!ReadU64(body, &offset, &out->meta.shape[i])) return false;
    }

    if (!ReadString(body, &offset, &out->meta.dtype)) return false;
    if (out->meta.dtype.empty()) return false;

    if (!ReadU32(body, &offset, &out->meta.row_block)) return false;
    if (!ReadU32(body, &offset, &out->meta.col_block)) return false;
    if (out->meta.row_block == 0 || out->meta.col_block == 0) return false;

    return offset == body.size();
}

std::string EncodeLoadWeightsChunkBody(
    const LoadWeightsChunkMsgHeader& header,
    std::span<const std::uint8_t> chunk_view) {
    std::string body;
    body.reserve(4 + 4 + 4 + 8 + 4 + chunk_view.size());

    AppendI32(&body, header.expert_id);
    AppendI32(&body, header.worker_id);
    AppendI32(&body, static_cast<std::int32_t>(header.tensor_kind));
    AppendU64(&body, header.chunk_offset);
    AppendU32(&body, static_cast<std::uint32_t>(chunk_view.size()));

    if (!chunk_view.empty()) {
        body.append(
            reinterpret_cast<const char*>(chunk_view.data()),
            chunk_view.size());
    }

    return body;
}

bool DecodeLoadWeightsChunkBody(
    const std::string& body,
    LoadWeightsChunkMsgHeader* out_header,
    std::span<const std::uint8_t>* out_chunk_view) {
    if (out_header == nullptr || out_chunk_view == nullptr) return false;

    std::size_t offset = 0;
    if (!ReadI32(body, &offset, &out_header->expert_id)) return false;
    if (!ReadI32(body, &offset, &out_header->worker_id)) return false;
    if (!ReadTensorKind(body, &offset, &out_header->tensor_kind)) return false;
    if (!ReadU64(body, &offset, &out_header->chunk_offset)) return false;

    std::uint32_t chunk_size = 0;
    if (!ReadU32(body, &offset, &chunk_size)) return false;
    if (offset + chunk_size > body.size()) return false;

    *out_chunk_view = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(body.data() + offset),
        static_cast<std::size_t>(chunk_size));

    offset += chunk_size;
    return offset == body.size();
}

std::string EncodeLoadWeightsEndBody(const LoadWeightsEndMsg& msg) {
    std::string body;
    body.reserve(4 + 4 + 4);

    AppendI32(&body, msg.expert_id);
    AppendI32(&body, msg.worker_id);
    AppendI32(&body, static_cast<std::int32_t>(msg.tensor_kind));

    return body;
}

bool DecodeLoadWeightsEndBody(const std::string& body, LoadWeightsEndMsg* out) {
    if (out == nullptr) return false;

    std::size_t offset = 0;
    if (!ReadI32(body, &offset, &out->expert_id)) return false;
    if (!ReadI32(body, &offset, &out->worker_id)) return false;
    if (!ReadTensorKind(body, &offset, &out->tensor_kind)) return false;

    return offset == body.size();
}

}  // namespace common
