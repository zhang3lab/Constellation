#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/backend/activation_codec_v2.h"
#include "expert_node_v2/backend/cpu/down_cpu_v2.h"
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

    const int inter_dim = 2048;
    const int hidden_dim = 7168;

    std::vector<float> h_host(inter_dim);
    for (int i = 0; i < inter_dim; ++i) {
        h_host[i] = std::sin(0.001f * static_cast<float>(i));
    }

    std::vector<std::uint8_t> y_ref_bytes;
    if (!RunDownReferenceV2(
            host_view.w_down,
            h_host.data(),
            common::ActivationDType::FP16,
            &y_ref_bytes)) {
        std::printf("RunDownReferenceV2 failed\n");
        return 1;
    }

    std::vector<std::uint8_t> y_cpu_bytes(
        static_cast<std::size_t>(hidden_dim) * sizeof(std::uint16_t), 0);
    if (!RunDownCpuV2(
            host_view.w_down,
            h_host.data(),
            y_cpu_bytes.data(),
            common::ActivationDType::FP16)) {
        std::printf("RunDownCpuV2 failed\n");
        return 1;
    }

    const auto* y_ref_u16 =
        reinterpret_cast<const std::uint16_t*>(y_ref_bytes.data());
    const auto* y_cpu_u16 =
        reinterpret_cast<const std::uint16_t*>(y_cpu_bytes.data());

    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    float dot = 0.0f;
    float norm_ref = 0.0f;
    float norm_cpu = 0.0f;

    for (int i = 0; i < hidden_dim; ++i) {
        const float ref_v = DecodeActivationToFloatV2(
            common::ActivationDType::FP16, y_ref_u16[i]);
        const float cpu_v = DecodeActivationToFloatV2(
            common::ActivationDType::FP16, y_cpu_u16[i]);
        const float abs_err = std::fabs(ref_v - cpu_v);

        if (abs_err > max_abs) max_abs = abs_err;
        sum_abs += abs_err;
        dot += ref_v * cpu_v;
        norm_ref += ref_v * ref_v;
        norm_cpu += cpu_v * cpu_v;
    }

    const float mean_abs = sum_abs / static_cast<float>(hidden_dim);
    const float cos =
        (norm_ref > 0.0f && norm_cpu > 0.0f)
            ? (dot / (std::sqrt(norm_ref) * std::sqrt(norm_cpu)))
            : 0.0f;

    std::printf(
        "cpu down compare: max_abs=%g mean_abs=%g cos=%g\n",
        max_abs, mean_abs, cos);

    for (int i = 0; i < 8; ++i) {
        const float ref_v = DecodeActivationToFloatV2(
            common::ActivationDType::FP16, y_ref_u16[i]);
        const float cpu_v = DecodeActivationToFloatV2(
            common::ActivationDType::FP16, y_cpu_u16[i]);
        std::printf("ref[%d]=%g cpu[%d]=%g\n", i, ref_v, i, cpu_v);
    }

    return 0;
}
