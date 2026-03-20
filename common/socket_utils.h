#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <sys/types.h>

#include "common/protocol.h"

namespace common {

bool SendAll(int fd, const void* data, std::size_t len);
bool RecvAll(int fd, void* data, std::size_t len);

bool SendBytes(int fd, const std::string& data);
bool SendMessage(int fd, const MsgHeader& hdr, const std::string& body);

}  // namespace common
