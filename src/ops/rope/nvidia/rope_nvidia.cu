#include "rope_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <cmath>

namespace {
template <typename T>
__global__ void ropeKernel(T *out, const T *in, const int64_t *pos, size_t seqlen, size_t nhead, size_t d, float theta) {
    const size_t half = d / 2;
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = seqlen * nhead * half;
    if (flat >= total) {
        return;
    }

    const size_t pair = flat % half;
    const size_t head_flat = flat / half;
    const size_t head = head_flat % nhead;
    const size_t seq = head_flat / nhead;
    const size_t base = (seq * nhead + head) * d;

    const float position = static_cast<float>(pos[seq]);
    const float angle = position / powf(theta, (2.0f * static_cast<float>(pair)) / static_cast<float>(d));
    const float s = sinf(angle);
    const float c = cosf(angle);
    const float a = llaisys::device::nvidia::toFloat(in[base + pair]);
    const float b = llaisys::device::nvidia::toFloat(in[base + pair + half]);
    out[base + pair] = llaisys::device::nvidia::fromFloat<T>(a * c - b * s);
    out[base + pair + half] = llaisys::device::nvidia::fromFloat<T>(b * c + a * s);
}

template <typename T>
void launchRope(llaisys::tensor_t out, llaisys::tensor_t in, llaisys::tensor_t pos_ids, float theta) {
    constexpr int kBlockSize = 256;
    const size_t seqlen = out->shape()[0];
    const size_t nhead = out->shape()[1];
    const size_t d = out->shape()[2];
    const size_t total = seqlen * nhead * (d / 2);
    ropeKernel<<<llaisys::device::nvidia::launch1D(total, kBlockSize), kBlockSize>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const T *>(in->data()),
        reinterpret_cast<const int64_t *>(pos_ids->data()),
        seqlen,
        nhead,
        d,
        theta);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

namespace llaisys::ops::nvidia {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return launchRope<float>(out, in, pos_ids, theta);
    case LLAISYS_DTYPE_F16:
        return launchRope<fp16_t>(out, in, pos_ids, theta);
    case LLAISYS_DTYPE_BF16:
        return launchRope<bf16_t>(out, in, pos_ids, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops::nvidia
