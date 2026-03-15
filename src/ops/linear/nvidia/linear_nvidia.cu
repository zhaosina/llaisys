#include "linear_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

namespace {
inline void checkCublas(cublasStatus_t status, const char *expr, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuBLAS error at ") + file + ":" + std::to_string(line) + " for " + expr);
    }
}

#define CUBLAS_CHECK(EXPR__) ::checkCublas((EXPR__), #EXPR__, __FILE__, __LINE__)

struct ScopedCublasHandle {
    cublasHandle_t handle;

    ScopedCublasHandle() : handle(nullptr) {
        CUBLAS_CHECK(cublasCreate(&handle));
    }

    ~ScopedCublasHandle() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }
};

cudaDataType_t cublasDataTypeFor(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t rows, size_t cols) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = rows * cols;
    if (flat >= total) {
        return;
    }
    const size_t col = flat % cols;
    const float value = llaisys::device::nvidia::toFloat(out[flat]) + llaisys::device::nvidia::toFloat(bias[col]);
    out[flat] = llaisys::device::nvidia::fromFloat<T>(value);
}

template <typename T>
void launchBiasAdd(llaisys::tensor_t out, llaisys::tensor_t bias) {
    constexpr int kBlockSize = 256;
    addBiasKernel<<<llaisys::device::nvidia::launch1D(out->numel(), kBlockSize), kBlockSize>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const T *>(bias->data()),
        out->shape()[0],
        out->shape()[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launchBias(llaisys::tensor_t out, llaisys::tensor_t bias) {
    return launchBiasAdd<T>(out, bias);
}
} // namespace

namespace llaisys::ops::nvidia {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    const int m = static_cast<int>(in->shape()[0]);
    const int n = static_cast<int>(weight->shape()[0]);
    const int k = static_cast<int>(weight->shape()[1]);

    ScopedCublasHandle handle;

    const auto data_type = cublasDataTypeFor(out->dtype());
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        weight->data(),
        data_type,
        k,
        in->data(),
        data_type,
        k,
        &beta,
        out->data(),
        data_type,
        n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));

    if (bias == nullptr) {
        return;
    }

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return launchBias<float>(out, bias);
    case LLAISYS_DTYPE_F16:
        return launchBias<fp16_t>(out, bias);
    case LLAISYS_DTYPE_BF16:
        return launchBias<bf16_t>(out, bias);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops::nvidia
