#include "common/header_codec.h"

#include <cstddef>
#include <cstdint>

#include "common/protocol.h"

namespace common {

std::string EncodeHeader(const MsgHeader& hdr) {
    std::string out;
    out.reserve(16);

    AppendU32(&out, hdr.magic);
    AppendU16(&out, hdr.version);
    AppendU16(&out, hdr.msg_type);
    AppendU32(&out, hdr.request_id);
    AppendU32(&out, hdr.body_len);

    return out;
}

bool DecodeHeader(const void* data, std::size_t len, MsgHeader* out) {
    if (len != 16 || out == nullptr) {
        return false;
    }

    const std::uint8_t* p = static_cast<const std::uint8_t*>(data);

    auto read_u16 = [](const std::uint8_t* q) -> std::uint16_t {
        return static_cast<std::uint16_t>(q[0]) |
               (static_cast<std::uint16_t>(q[1]) << 8);
    };

    auto read_u32 = [](const std::uint8_t* q) -> std::uint32_t {
        return static_cast<std::uint32_t>(q[0]) |
               (static_cast<std::uint32_t>(q[1]) << 8) |
               (static_cast<std::uint32_t>(q[2]) << 16) |
               (static_cast<std::uint32_t>(q[3]) << 24);
    };

    out->magic = read_u32(p + 0);
    out->version = read_u16(p + 4);
    out->msg_type = read_u16(p + 6);
    out->request_id = read_u32(p + 8);
    out->body_len = read_u32(p + 12);
    return true;
}

}  // namespace common
