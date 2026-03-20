#include "common/socket_utils.h"

#include <cerrno>
#include <sys/socket.h>
#include <unistd.h>

#include "common/header_codec.h"

namespace common {

bool SendAll(int fd, const void* data, std::size_t len) {
    const char* p = static_cast<const char*>(data);
    std::size_t remaining = len;

    while (remaining > 0) {
        ssize_t n = ::send(fd, p, remaining, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        p += n;
        remaining -= static_cast<std::size_t>(n);
    }
    return true;
}

bool RecvAll(int fd, void* data, std::size_t len) {
    char* p = static_cast<char*>(data);
    std::size_t remaining = len;

    while (remaining > 0) {
        ssize_t n = ::recv(fd, p, remaining, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        p += n;
        remaining -= static_cast<std::size_t>(n);
    }
    return true;
}

bool SendBytes(int fd, const std::string& data) {
    return SendAll(fd, data.data(), data.size());
}

bool SendMessage(int fd, const MsgHeader& hdr, const std::string& body) {
    std::string hdr_bytes = EncodeHeader(hdr);
    return SendBytes(fd, hdr_bytes) && SendBytes(fd, body);
}

}  // namespace common
