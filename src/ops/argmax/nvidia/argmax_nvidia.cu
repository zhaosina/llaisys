#include "argmax_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <vector>

namespace {
template <typename T>
void argmaxHost(int64_t *out_idx, T *out_val, const T *vals, size_t n) {
    if (n == 0) {
        *out_idx = 0;
        *out_val = llaisys::device::nvidia::fromFloat<T>(0.0f);
        return;
    }
    float max_value = llaisys::device::nvidia::toFloat(vals[0]);
    int64_t max_index = 0;
    for (size_t i = 1; i < n; i++) {
        const float value = llaisys::device::nvidia::toFloat(vals[i]);
        if (value > max_value) {
            max_value = value;
            max_index = static_cast<int64_t>(i);
        }
    }
    *out_idx = max_index;
    *out_val = llaisys::device::nvidia::fromFloat<T>(max_value);
}

template <typename T>
void launchArgmax(llaisys::tensor_t max_idx, llaisys::tensor_t max_val, llaisys::tensor_t vals) {
    std::vector<T> host_vals(vals->numel());
    T host_val{};
    int64_t host_idx = 0;

    CUDA_CHECK(cudaMemcpy(host_vals.data(), vals->data(), vals->numel() * sizeof(T), cudaMemcpyDeviceToHost));
    argmaxHost(&host_idx, &host_val, host_vals.data(), host_vals.size());
    CUDA_CHECK(cudaMemcpy(max_idx->data(), &host_idx, sizeof(host_idx), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(max_val->data(), &host_val, sizeof(host_val), cudaMemcpyHostToDevice));
}
} // namespace

namespace llaisys::ops::nvidia {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        return launchArgmax<float>(max_idx, max_val, vals);
    case LLAISYS_DTYPE_F16:
        return launchArgmax<fp16_t>(max_idx, max_val, vals);
    case LLAISYS_DTYPE_BF16:
        return launchArgmax<bf16_t>(max_idx, max_val, vals);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}
} // namespace llaisys::ops::nvidia
