#pragma once

#include <cstddef>
#include <string>

#include "common/protocol.h"

namespace common {

// Encodes MsgHeader into the 16-byte little-endian wire format.
std::string EncodeHeader(const MsgHeader& hdr);

// Decodes a 16-byte little-endian wire-format header into MsgHeader.
bool DecodeHeader(const void* data, std::size_t len, MsgHeader* out);

}  // namespace common
