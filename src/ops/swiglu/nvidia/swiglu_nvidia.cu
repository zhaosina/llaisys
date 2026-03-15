#include "swiglu_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <cmath>

namespace {
template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    const float g = llaisys::device::nvidia::toFloat(gate[idx]);
    const float u = llaisys::device::nvidia::toFloat(up[idx]);
    const float value = u * (g / (1.0f + expf(-g)));
    out[idx] = llaisys::device::nvidia::fromFloat<T>(value);
}

template <typename T>
void launchSwiglu(llaisys::tensor_t out, llaisys::tensor_t gate, llaisys::tensor_t up) {
    constexpr int kBlockSize = 256;
    swigluKernel<<<llaisys::device::nvidia::launch1D(out->numel(), kBlockSize), kBlockSize>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const T *>(gate->data()),
        reinterpret_cast<const T *>(up->data()),
        out->numel());
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

namespace llaisys::ops::nvidia {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return launchSwiglu<float>(out, gate, up);
    case LLAISYS_DTYPE_F16:
        return launchSwiglu<fp16_t>(out, gate, up);
    case LLAISYS_DTYPE_BF16:
        return launchSwiglu<bf16_t>(out, gate, up);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops::nvidia
