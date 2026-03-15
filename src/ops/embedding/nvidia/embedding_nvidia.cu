#include "embedding_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

namespace {
template <typename T>
__global__ void embeddingKernel(T *out, const int64_t *index, const T *weight, size_t nindex, size_t dim) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = nindex * dim;
    if (flat >= total) {
        return;
    }
    const size_t row = flat / dim;
    const size_t col = flat % dim;
    const size_t weight_row = static_cast<size_t>(index[row]);
    out[flat] = weight[weight_row * dim + col];
}

template <typename T>
void launchEmbedding(llaisys::tensor_t out, llaisys::tensor_t index, llaisys::tensor_t weight) {
    constexpr int kBlockSize = 256;
    const size_t total = out->numel();
    embeddingKernel<<<llaisys::device::nvidia::launch1D(total, kBlockSize), kBlockSize>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const int64_t *>(index->data()),
        reinterpret_cast<const T *>(weight->data()),
        index->shape()[0],
        weight->shape()[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

namespace llaisys::ops::nvidia {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return launchEmbedding<float>(out, index, weight);
    case LLAISYS_DTYPE_F16:
        return launchEmbedding<fp16_t>(out, index, weight);
    case LLAISYS_DTYPE_BF16:
        return launchEmbedding<bf16_t>(out, index, weight);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops::nvidia
