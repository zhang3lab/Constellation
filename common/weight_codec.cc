#include "common/weight_codec.h"

#include <cstddef>
#include <cstdint>

#include "common/protocol.h"

namespace common {
namespace {

bool ReadU32(const std::string& buf, std::size_t* offset, std::uint32_t* out) {
    if (*offset + 4 > buf.size()) return false;
    const std::uint8_t* p =
        reinterpret_cast<const std::uint8_t*>(buf.data() + *offset);
    *out = static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
    *offset += 4;
    return true;
}

bool ReadI32(const std::string& buf, std::size_t* offset, std::int32_t* out) {
    std::uint32_t tmp = 0;
    if (!ReadU32(buf, offset, &tmp)) return false;
    *out = static_cast<std::int32_t>(tmp);
    return true;
}

bool ReadU64(const std::string& buf, std::size_t* offset, std::uint64_t* out) {
    if (*offset + 8 > buf.size()) return false;
    const std::uint8_t* p =
        reinterpret_cast<const std::uint8_t*>(buf.data() + *offset);
    *out = static_cast<std::uint64_t>(p[0]) |
           (static_cast<std::uint64_t>(p[1]) << 8) |
           (static_cast<std::uint64_t>(p[2]) << 16) |
           (static_cast<std::uint64_t>(p[3]) << 24) |
           (static_cast<std::uint64_t>(p[4]) << 32) |
           (static_cast<std::uint64_t>(p[5]) << 40) |
           (static_cast<std::uint64_t>(p[6]) << 48) |
           (static_cast<std::uint64_t>(p[7]) << 56);
    *offset += 8;
    return true;
}

}  // namespace

std::string EncodeLoadWeightsBeginBody(const LoadWeightsBeginMsg& msg) {
    std::string body;
    body.reserve(4 + 4 + 4 + 8);

    AppendI32(&body, msg.expert_id);
    AppendI32(&body, msg.local_gpu_id);
    AppendI32(&body, static_cast<std::int32_t>(msg.tensor_kind));
    AppendU64(&body, msg.total_bytes);

    return body;
}

bool DecodeLoadWeightsBeginBody(
    const std::string& body,
    LoadWeightsBeginMsg* out) {
    if (out == nullptr) return false;

    std::size_t offset = 0;
    if (!ReadI32(body, &offset, &out->expert_id)) return false;
    if (!ReadI32(body, &offset, &out->local_gpu_id)) return false;

    std::int32_t tensor_kind_raw = -1;
    if (!ReadI32(body, &offset, &tensor_kind_raw)) return false;
    if (!ReadU64(body, &offset, &out->total_bytes)) return false;

    if (tensor_kind_raw < static_cast<std::int32_t>(TensorKind::WUp) ||
        tensor_kind_raw > static_cast<std::int32_t>(TensorKind::WDown)) {
        return false;
    }
    out->tensor_kind = static_cast<TensorKind>(tensor_kind_raw);

    return offset == body.size();
}

}  // namespace common
