#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/backend/cpu/fused_up_gate_cpu_v2.h"
#include "expert_node_v2/backend/dummy_expert_data_v2.h"
#include "expert_node_v2/backend/expert_reference_v2.h"
#include "expert_node_v2/expert_format_v2.h"

int main() {
    ExpertTensorBundleV2 bundle;
    FillDummyExpertBundleV2(&bundle);

    ExpertWeightsViewV2 host_view;
    if (!BuildExpertWeightsViewV2(bundle, &host_view)) {
        std::printf("BuildExpertWeightsViewV2 failed\n");
        return 1;
    }

    const int hidden_dim = 7168;
    const int inter_dim = 2048;

    std::vector<float> x_float;
    std::vector<std::uint16_t> x_fp16;
    FillDummyInputActivationV2(
        hidden_dim,
        common::ActivationDType::FP16,
        &x_float,
        &x_fp16);

    std::vector<float> h_ref;
    if (!RunFusedUpGateReferenceV2(
            host_view.w_up,
            host_view.w_gate,
            x_fp16.data(),
            common::ActivationDType::FP16,
            &h_ref)) {
        std::printf("RunFusedUpGateReferenceV2 failed\n");
        return 1;
    }

    std::vector<float> h_cpu(inter_dim, 0.0f);
    if (!RunFusedUpGateCpuV2(
            host_view.w_up,
            host_view.w_gate,
            x_fp16.data(),
            common::ActivationDType::FP16,
            h_cpu.data())) {
        std::printf("RunFusedUpGateCpuV2 failed\n");
        return 1;
    }

    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    float dot = 0.0f;
    float norm_ref = 0.0f;
    float norm_cpu = 0.0f;

    for (int i = 0; i < inter_dim; ++i) {
        const float ref_v = h_ref[i];
        const float cpu_v = h_cpu[i];
        const float abs_err = std::fabs(ref_v - cpu_v);

        if (abs_err > max_abs) max_abs = abs_err;
        sum_abs += abs_err;
        dot += ref_v * cpu_v;
        norm_ref += ref_v * ref_v;
        norm_cpu += cpu_v * cpu_v;
    }

    const float mean_abs = sum_abs / static_cast<float>(inter_dim);
    const float cos =
        (norm_ref > 0.0f && norm_cpu > 0.0f)
            ? (dot / (std::sqrt(norm_ref) * std::sqrt(norm_cpu)))
            : 0.0f;

    std::printf(
        "cpu fused up/gate compare: max_abs=%g mean_abs=%g cos=%g\n",
        max_abs, mean_abs, cos);

    for (int i = 0; i < 8; ++i) {
        std::printf("ref[%d]=%g cpu[%d]=%g\n", i, h_ref[i], i, h_cpu[i]);
    }

    return 0;
}
