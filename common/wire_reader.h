#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "common/protocol.h"

namespace common {

bool ReadU32(const std::string& buf, std::size_t* offset, std::uint32_t* out);
bool ReadI32(const std::string& buf, std::size_t* offset, std::int32_t* out);
bool ReadU64(const std::string& buf, std::size_t* offset, std::uint64_t* out);
bool ReadBytes(const std::string& buf, std::size_t* offset, std::size_t n, std::string* out);
bool ReadTensorKind(const std::string& buf, std::size_t* offset, TensorKind* out);

}  // namespace common
