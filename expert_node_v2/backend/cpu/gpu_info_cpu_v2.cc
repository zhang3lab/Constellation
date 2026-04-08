#include "expert_node_v2/backend/cpu/gpu_info_cpu_v2.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace expert_node_v2 {
namespace {

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

bool QueryGpuInfoCpuV2(
    int num_cpu_workers,
    std::vector<common::StaticGpuInfo>* out_gpus) {
    if (out_gpus == nullptr) return false;
    out_gpus->clear();

    if (num_cpu_workers <= 0) {
        return false;
    }

    std::uint64_t total_bytes = 0;
    std::uint64_t free_bytes = 0;
    if (!ReadHostMemoryInfo(&total_bytes, &free_bytes)) {
        return false;
    }

    std::string cpu_name = "cpu";
    {
        std::string detected_name;
        if (ReadCpuName(&detected_name) && !detected_name.empty()) {
            cpu_name = detected_name;
        }
    }

    const std::uint64_t per_worker_total =
        total_bytes / static_cast<std::uint64_t>(num_cpu_workers);
    const std::uint64_t per_worker_free =
        free_bytes / static_cast<std::uint64_t>(num_cpu_workers);

    out_gpus->reserve(static_cast<std::size_t>(num_cpu_workers));
    for (int i = 0; i < num_cpu_workers; ++i) {
        common::StaticGpuInfo gpu;
        gpu.worker_id = i;
        gpu.gpu_vendor = common::GpuVendor::Cpu;
        gpu.name = cpu_name;
        gpu.total_bytes = per_worker_total;
        gpu.free_bytes = per_worker_free;
        out_gpus->push_back(gpu);
    }

    return true;
}

}  // namespace expert_node_v2
