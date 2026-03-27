#include "expert_node_v2/expert_format_v2.h"

#include <cstdio>

namespace {

unsigned long long dim0(const HostTensorV2& t) {
    return t.meta.shape.size() > 0
        ? static_cast<unsigned long long>(t.meta.shape[0])
        : 0ULL;
}

unsigned long long dim1(const HostTensorV2& t) {
    return t.meta.shape.size() > 1
        ? static_cast<unsigned long long>(t.meta.shape[1])
        : 0ULL;
}

void print_one_tensor(const char* prefix, const char* name, const HostTensorV2& t) {
    std::fprintf(stderr,
                 "%s%s: ready=%d bytes=%zu shape=(%llu,%llu) dtype=%s block=(%u,%u)\n",
                 prefix,
                 name,
                 static_cast<int>(t.ready),
                 t.bytes.size(),
                 dim0(t),
                 dim1(t),
                 t.meta.dtype.c_str(),
                 static_cast<unsigned>(t.meta.row_block),
                 static_cast<unsigned>(t.meta.col_block));
}

}  // namespace

void ExpertTensorBundleV2::debug_print(const char* prefix) const {
    if (prefix == nullptr) {
        prefix = "";
    }

    std::fprintf(stderr,
                 "%sExpertTensorBundleV2 all_ready=%d\n",
                 prefix,
                 static_cast<int>(all_ready()));

    print_one_tensor(prefix, "w_up", w_up);
    print_one_tensor(prefix, "w_up_scale", w_up_scale);
    print_one_tensor(prefix, "w_gate", w_gate);
    print_one_tensor(prefix, "w_gate_scale", w_gate_scale);
    print_one_tensor(prefix, "w_down", w_down);
    print_one_tensor(prefix, "w_down_scale", w_down_scale);
}
