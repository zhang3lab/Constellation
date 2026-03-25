#include "common/protocol.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace common {
namespace {

inline void AppendByte(std::string* out, unsigned char b) {
    out->push_back(static_cast<char>(b));
}

}  // namespace

void AppendU16(std::string* out, std::uint16_t x) {
    AppendByte(out, static_cast<unsigned char>((x >> 0) & 0xFF));
    AppendByte(out, static_cast<unsigned char>((x >> 8) & 0xFF));
}

void AppendU32(std::string* out, std::uint32_t x) {
    AppendByte(out, static_cast<unsigned char>((x >> 0) & 0xFF));
    AppendByte(out, static_cast<unsigned char>((x >> 8) & 0xFF));
    AppendByte(out, static_cast<unsigned char>((x >> 16) & 0xFF));
    AppendByte(out, static_cast<unsigned char>((x >> 24) & 0xFF));
}

void AppendI32(std::string* out, std::int32_t x) {
    AppendU32(out, static_cast<std::uint32_t>(x));
}

void AppendU64(std::string* out, std::uint64_t x) {
    AppendByte(out, static_cast<unsigned char>((x >> 0) & 0xFF));
    AppendByte(out, static_cast<unsigned char>((x >> 8) & 0xFF));
    AppendByte(out, static_cast<unsigned char>((x >> 16) & 0xFF));
    AppendByte(out, static_cast<unsigned char>((x >> 24) & 0xFF));
    AppendByte(out, static_cast<unsigned char>((x >> 32) & 0xFF));
    AppendByte(out, static_cast<unsigned char>((x >> 40) & 0xFF));
    AppendByte(out, static_cast<unsigned char>((x >> 48) & 0xFF));
    AppendByte(out, static_cast<unsigned char>((x >> 56) & 0xFF));
}

void AppendString(std::string* out, const std::string& s) {
    if (s.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("string too long for u32 length prefix");
    }
    AppendU32(out, static_cast<std::uint32_t>(s.size()));
    out->append(s);
}

bool ReadU16(const std::string& buf, std::size_t* offset, std::uint16_t* out) {
    if (offset == nullptr || out == nullptr) return false;
    if (*offset + sizeof(std::uint16_t) > buf.size()) return false;
    std::memcpy(out, buf.data() + *offset, sizeof(std::uint16_t));
    *offset += sizeof(std::uint16_t);
    return true;
}

bool ReadU32(const std::string& buf, std::size_t* offset, std::uint32_t* out) {
    if (offset == nullptr || out == nullptr) return false;
    if (*offset + sizeof(std::uint32_t) > buf.size()) return false;
    std::memcpy(out, buf.data() + *offset, sizeof(std::uint32_t));
    *offset += sizeof(std::uint32_t);
    return true;
}

bool ReadI32(const std::string& buf, std::size_t* offset, std::int32_t* out) {
    if (offset == nullptr || out == nullptr) return false;
    if (*offset + sizeof(std::int32_t) > buf.size()) return false;
    std::memcpy(out, buf.data() + *offset, sizeof(std::int32_t));
    *offset += sizeof(std::int32_t);
    return true;
}

bool ReadU64(const std::string& buf, std::size_t* offset, std::uint64_t* out) {
    if (offset == nullptr || out == nullptr) return false;
    if (*offset + sizeof(std::uint64_t) > buf.size()) return false;
    std::memcpy(out, buf.data() + *offset, sizeof(std::uint64_t));
    *offset += sizeof(std::uint64_t);
    return true;
}

bool ReadBytes(const std::string& buf, std::size_t* offset, std::size_t n, std::string* out) {
    if (offset == nullptr || out == nullptr) return false;
    if (*offset + n > buf.size()) return false;
    out->assign(buf.data() + *offset, n);
    *offset += n;
    return true;
}

bool ReadString(const std::string& buf, std::size_t* offset, std::string* out) {
    if (offset == nullptr || out == nullptr) return false;

    std::uint32_t n = 0;
    if (!ReadU32(buf, offset, &n)) return false;
    return ReadBytes(buf, offset, static_cast<std::size_t>(n), out);
}

bool ReadTensorKind(const std::string& buf, std::size_t* offset, TensorKind* out) {
    if (out == nullptr) return false;

    std::int32_t raw = -1;
    if (!ReadI32(buf, offset, &raw)) return false;

    if (raw < static_cast<std::int32_t>(TensorKind::WUp) ||
        raw > static_cast<std::int32_t>(TensorKind::WDownScale)) {
        return false;
    }

    *out = static_cast<TensorKind>(raw);
    return true;
}

}  // namespace common
