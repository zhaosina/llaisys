#include "add_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

namespace {
template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t numel) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    c[idx] = llaisys::device::nvidia::fromFloat<T>(
        llaisys::device::nvidia::toFloat(a[idx]) + llaisys::device::nvidia::toFloat(b[idx]));
}

template <typename T>
void launchAdd(T *c, const T *a, const T *b, size_t numel) {
    constexpr int kBlockSize = 256;
    addKernel<<<llaisys::device::nvidia::launch1D(numel, kBlockSize), kBlockSize>>>(c, a, b, numel);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

namespace llaisys::ops::nvidia {
void add(tensor_t c, tensor_t a, tensor_t b) {
    switch (c->dtype()) {
    case LLAISYS_DTYPE_F32:
        return launchAdd(reinterpret_cast<float *>(c->data()),
                         reinterpret_cast<const float *>(a->data()),
                         reinterpret_cast<const float *>(b->data()),
                         c->numel());
    case LLAISYS_DTYPE_F16:
        return launchAdd(reinterpret_cast<fp16_t *>(c->data()),
                         reinterpret_cast<const fp16_t *>(a->data()),
                         reinterpret_cast<const fp16_t *>(b->data()),
                         c->numel());
    case LLAISYS_DTYPE_BF16:
        return launchAdd(reinterpret_cast<bf16_t *>(c->data()),
                         reinterpret_cast<const bf16_t *>(a->data()),
                         reinterpret_cast<const bf16_t *>(b->data()),
                         c->numel());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(c->dtype());
    }
}
} // namespace llaisys::ops::nvidia
