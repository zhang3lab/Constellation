#include "common/wire_reader.h"

namespace common {

bool ReadU32(const std::string& buf, std::size_t* offset, std::uint32_t* out) {
    if (offset == nullptr || out == nullptr) return false;
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
    if (offset == nullptr || out == nullptr) return false;
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

bool ReadBytes(const std::string& buf, std::size_t* offset, std::size_t n, std::string* out) {
    if (offset == nullptr || out == nullptr) return false;
    if (*offset + n > buf.size()) return false;

    out->assign(buf.data() + *offset, n);
    *offset += n;
    return true;
}

bool ReadTensorKind(const std::string& buf, std::size_t* offset, TensorKind* out) {
    if (out == nullptr) return false;

    std::int32_t raw = -1;
    if (!ReadI32(buf, offset, &raw)) return false;

    if (raw < static_cast<std::int32_t>(TensorKind::WUp) ||
        raw > static_cast<std::int32_t>(TensorKind::WDown)) {
        return false;
    }

    *out = static_cast<TensorKind>(raw);
    return true;
}

}  // namespace common
