#include "expert_node_v2/backend/cpu/gpu_info_cpu_v2.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace {

constexpr int kLocalCpuWorkerCount = 1;

std::uint32_t make_cpu_capability_flags() {
    std::uint32_t flags = 0;
    flags |= common::kGpuCapFp16;
    flags |= common::kGpuCapBf16;
    return flags;
}

std::string make_cpu_arch_name() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    return "arm64";
#else
    return "cpu";
#endif
}

common::GpuStatus make_initial_gpu_status() {
    return common::GpuStatus::Idle;
}

#if defined(__linux__)
bool ReadMemoryInfoLinuxProcMeminfo(
    std::uint64_t* out_total_bytes,
    std::uint64_t* out_free_bytes) {
    if (out_total_bytes == nullptr || out_free_bytes == nullptr) return false;

    FILE* f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) return false;

    char key[64];
    unsigned long long value_kb = 0;
    char unit[32];

    bool found_total = false;
    bool found_available = false;
    std::uint64_t total_bytes = 0;
    std::uint64_t free_bytes = 0;

    while (std::fscanf(f, "%63s %llu %31s", key, &value_kb, unit) == 3) {
        if (std::strcmp(key, "MemTotal:") == 0) {
            total_bytes = static_cast<std::uint64_t>(value_kb) * 1024ull;
            found_total = true;
        } else if (std::strcmp(key, "MemAvailable:") == 0) {
            free_bytes = static_cast<std::uint64_t>(value_kb) * 1024ull;
            found_available = true;
        }

        if (found_total && found_available) {
            break;
        }
    }

    std::fclose(f);

    if (!found_total || !found_available) {
        return false;
    }

    if (free_bytes > total_bytes) {
        free_bytes = total_bytes;
    }

    *out_total_bytes = total_bytes;
    *out_free_bytes = free_bytes;
    return true;
}

bool ReadMemoryInfoLinuxSysinfo(
    std::uint64_t* out_total_bytes,
    std::uint64_t* out_free_bytes) {
    if (out_total_bytes == nullptr || out_free_bytes == nullptr) return false;

    struct sysinfo info {};
    if (sysinfo(&info) != 0) {
        return false;
    }

    const std::uint64_t unit = static_cast<std::uint64_t>(info.mem_unit);
    std::uint64_t total_bytes =
        static_cast<std::uint64_t>(info.totalram) * unit;
    std::uint64_t free_bytes =
        static_cast<std::uint64_t>(info.freeram) * unit;

    if (free_bytes > total_bytes) {
        free_bytes = total_bytes;
    }

    *out_total_bytes = total_bytes;
    *out_free_bytes = free_bytes;
    return true;
}

bool ReadCpuNameLinux(std::string* out_name) {
    if (out_name == nullptr) return false;

    FILE* f = std::fopen("/proc/cpuinfo", "r");
    if (f == nullptr) return false;

    char line[512];
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        const char* key = "model name";
        const std::size_t key_len = std::strlen(key);
        if (std::strncmp(line, key, key_len) != 0) {
            continue;
        }

        const char* colon = std::strchr(line, ':');
        if (colon == nullptr) {
            continue;
        }

        std::string name = colon + 1;
        while (!name.empty() && (name.front() == ' ' || name.front() == '\t')) {
            name.erase(name.begin());
        }
        while (!name.empty() &&
               (name.back() == '\n' || name.back() == '\r' ||
                name.back() == ' ' || name.back() == '\t')) {
            name.pop_back();
        }

        std::fclose(f);
        if (name.empty()) return false;
        *out_name = name;
        return true;
    }

    std::fclose(f);
    return false;
}
#endif

#if defined(__APPLE__)
bool ReadMemoryInfoMac(
    std::uint64_t* out_total_bytes,
    std::uint64_t* out_free_bytes) {
    if (out_total_bytes == nullptr || out_free_bytes == nullptr) return false;

    std::uint64_t memsize = 0;
    size_t memsize_len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &memsize_len, nullptr, 0) != 0) {
        return false;
    }

    mach_port_t host = mach_host_self();

    vm_size_t page_size = 0;
    if (host_page_size(host, &page_size) != KERN_SUCCESS) {
        mach_port_deallocate(mach_task_self(), host);
        return false;
    }

    vm_statistics64_data_t vm_stat {};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(
            host,
            HOST_VM_INFO64,
            reinterpret_cast<host_info64_t>(&vm_stat),
            &count) != KERN_SUCCESS) {
        mach_port_deallocate(mach_task_self(), host);
        return false;
    }

    mach_port_deallocate(mach_task_self(), host);

    const std::uint64_t reclaimable_pages =
        static_cast<std::uint64_t>(vm_stat.free_count) +
        static_cast<std::uint64_t>(vm_stat.inactive_count) +
        static_cast<std::uint64_t>(vm_stat.speculative_count);

    std::uint64_t free_bytes =
        reclaimable_pages * static_cast<std::uint64_t>(page_size);
    if (free_bytes > memsize) {
        free_bytes = memsize;
    }

    *out_total_bytes = memsize;
    *out_free_bytes = free_bytes;
    return true;
}

bool ReadCpuNameMac(std::string* out_name) {
    if (out_name == nullptr) return false;

    size_t len = 0;
    if (sysctlbyname("machdep.cpu.brand_string", nullptr, &len, nullptr, 0) != 0) {
        return false;
    }
    if (len == 0) return false;

    std::string buf(len, '\0');
    if (sysctlbyname("machdep.cpu.brand_string", buf.data(), &len, nullptr, 0) != 0) {
        return false;
    }

    while (!buf.empty() &&
           (buf.back() == '\0' || buf.back() == '\n' || buf.back() == '\r')) {
        buf.pop_back();
    }
    if (buf.empty()) return false;

    *out_name = buf;
    return true;
}
#endif

bool ReadHostMemoryInfo(
    std::uint64_t* out_total_bytes,
    std::uint64_t* out_free_bytes) {
    if (out_total_bytes == nullptr || out_free_bytes == nullptr) return false;

#if defined(__linux__)
    if (ReadMemoryInfoLinuxProcMeminfo(out_total_bytes, out_free_bytes)) {
        return true;
    }
    if (ReadMemoryInfoLinuxSysinfo(out_total_bytes, out_free_bytes)) {
        return true;
    }
    return false;
#elif defined(__APPLE__)
    return ReadMemoryInfoMac(out_total_bytes, out_free_bytes);
#else
    return false;
#endif
}

bool ReadCpuName(std::string* out_name) {
    if (out_name == nullptr) return false;

#if defined(__linux__)
    return ReadCpuNameLinux(out_name);
#elif defined(__APPLE__)
    return ReadCpuNameMac(out_name);
#else
    return false;
#endif
}

}  // namespace

bool BuildLocalCpuGpuInfosV2(
    std::int32_t worker_id_begin,
    std::uint32_t worker_port_base,
    std::vector<common::StaticGpuInfo>* out) {
    if (out == nullptr) return false;
    out->clear();

    std::uint64_t total_mem_bytes = 0;
    std::uint64_t free_mem_bytes = 0;
    if (!ReadHostMemoryInfo(&total_mem_bytes, &free_mem_bytes)) {
        return false;
    }
    (void)free_mem_bytes;

    std::string cpu_name = "cpu";
    {
        std::string detected_name;
        if (ReadCpuName(&detected_name) && !detected_name.empty()) {
            cpu_name = detected_name;
        }
    }

    for (int i = 0; i < kLocalCpuWorkerCount; ++i) {
        common::StaticGpuInfo gpu;
        gpu.worker_id = worker_id_begin + i;
        gpu.gpu_name = cpu_name;
        gpu.total_mem_bytes =
            total_mem_bytes / static_cast<std::uint64_t>(kLocalCpuWorkerCount);
        gpu.worker_port =
            worker_port_base + static_cast<std::uint32_t>(gpu.worker_id);

        gpu.gpu_vendor = common::GpuVendor::Cpu;
        gpu.capability_flags = make_cpu_capability_flags();
        gpu.arch_name = make_cpu_arch_name();

        out->push_back(std::move(gpu));
    }

    return true;
}

bool BuildLocalCpuDynamicGpuInfosV2(
    std::int32_t worker_id_begin,
    std::vector<common::DynamicGpuInfo>* out) {
    if (out == nullptr) return false;
    out->clear();

    std::uint64_t total_mem_bytes = 0;
    std::uint64_t free_mem_bytes = 0;
    if (!ReadHostMemoryInfo(&total_mem_bytes, &free_mem_bytes)) {
        return false;
    }
    (void)total_mem_bytes;

    for (int i = 0; i < kLocalCpuWorkerCount; ++i) {
        common::DynamicGpuInfo gpu;
        gpu.worker_id = worker_id_begin + i;
        gpu.free_mem_bytes =
            free_mem_bytes / static_cast<std::uint64_t>(kLocalCpuWorkerCount);
        gpu.gpu_status = make_initial_gpu_status();

        out->push_back(std::move(gpu));
    }

    return true;
}
