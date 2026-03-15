#include "rms_norm_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <cmath>

namespace {
template <typename T>
__global__ void rmsNormKernel(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const T *in_row = in + row * cols;
    T *out_row = out + row * cols;

    float sum = 0.0f;
    for (size_t col = 0; col < cols; col++) {
        const float value = llaisys::device::nvidia::toFloat(in_row[col]);
        sum += value * value;
    }
    const float inv = rsqrtf(sum / static_cast<float>(cols) + eps);
    for (size_t col = 0; col < cols; col++) {
        const float value = llaisys::device::nvidia::toFloat(in_row[col]);
        const float scale = llaisys::device::nvidia::toFloat(weight[col]);
        out_row[col] = llaisys::device::nvidia::fromFloat<T>(value * inv * scale);
    }
}

template <typename T>
__global__ void qwenRmsNormKernel(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const T *in_row = in + row * cols;
    T *out_row = out + row * cols;

    float sum = 0.0f;
    for (size_t col = 0; col < cols; col++) {
        const float value = llaisys::device::nvidia::toFloat(in_row[col]);
        sum += value * value;
    }
    const float inv = rsqrtf(sum / static_cast<float>(cols) + eps);
    for (size_t col = 0; col < cols; col++) {
        const float value = llaisys::device::nvidia::toFloat(in_row[col]);
        const float normalized = llaisys::device::nvidia::toFloat(llaisys::device::nvidia::fromFloat<T>(value * inv));
        const float scale = llaisys::device::nvidia::toFloat(weight[col]);
        out_row[col] = llaisys::device::nvidia::fromFloat<T>(normalized * scale);
    }
}

template <typename T, bool kQwenStyle>
void launchRmsNorm(llaisys::tensor_t out, llaisys::tensor_t in, llaisys::tensor_t weight, float eps) {
    constexpr int kBlockSize = 256;
    const size_t rows = in->shape()[0];
    const size_t cols = in->shape()[1];
    if constexpr (kQwenStyle) {
        qwenRmsNormKernel<<<llaisys::device::nvidia::launch1D(rows, kBlockSize), kBlockSize>>>(
            reinterpret_cast<T *>(out->data()),
            reinterpret_cast<const T *>(in->data()),
            reinterpret_cast<const T *>(weight->data()),
            rows,
            cols,
            eps);
    } else {
        rmsNormKernel<<<llaisys::device::nvidia::launch1D(rows, kBlockSize), kBlockSize>>>(
            reinterpret_cast<T *>(out->data()),
            reinterpret_cast<const T *>(in->data()),
            reinterpret_cast<const T *>(weight->data()),
            rows,
            cols,
            eps);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

namespace llaisys::ops::nvidia {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return launchRmsNorm<float, false>(out, in, weight, eps);
    case LLAISYS_DTYPE_F16:
        return launchRmsNorm<fp16_t, false>(out, in, weight, eps);
    case LLAISYS_DTYPE_BF16:
        return launchRmsNorm<bf16_t, false>(out, in, weight, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

void qwen_rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return launchRmsNorm<float, true>(out, in, weight, eps);
    case LLAISYS_DTYPE_F16:
        return launchRmsNorm<fp16_t, true>(out, in, weight, eps);
    case LLAISYS_DTYPE_BF16:
        return launchRmsNorm<bf16_t, true>(out, in, weight, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops::nvidia
