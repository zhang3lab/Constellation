#include "common/protocol.h"

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

}  // namespace common
