#pragma once

#include "llaisys.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace llaisys::device::nvidia {

inline void checkCuda(cudaError_t status, const char *expr, const char *file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + file + ":" + std::to_string(line) +
                                 " for " + expr + ": " + cudaGetErrorString(status));
    }
}

#define CUDA_CHECK(EXPR__) ::llaisys::device::nvidia::checkCuda((EXPR__), #EXPR__, __FILE__, __LINE__)

inline cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::runtime_error("Unsupported memcpy kind");
    }
}

__host__ __device__ inline float fp16ToFloat(llaisys::fp16_t value) {
    const uint16_t h = value._v;
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;

    uint32_t bits = 0;
    if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = -14;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        bits = sign | (static_cast<uint32_t>(exponent + 112) << 23) | (mantissa << 13);
    }

    union {
        uint32_t bits;
        float value;
    } out{bits};
    return out.value;
}

__host__ __device__ inline llaisys::fp16_t floatToFp16(float value) {
    union {
        uint32_t bits;
        float value;
    } in{0};
    in.value = value;

    const uint16_t sign = static_cast<uint16_t>((in.bits >> 16) & 0x8000u);
    const int32_t exponent = static_cast<int32_t>((in.bits >> 23) & 0xFFu) - 127;
    const uint32_t mantissa = in.bits & 0x7FFFFFu;

    if (exponent >= 16) {
        if (((in.bits >> 23) & 0xFFu) == 0xFFu && mantissa != 0) {
            return llaisys::fp16_t{static_cast<uint16_t>(sign | 0x7E00u)};
        }
        return llaisys::fp16_t{static_cast<uint16_t>(sign | 0x7C00u)};
    }
    if (exponent >= -14) {
        return llaisys::fp16_t{static_cast<uint16_t>(sign | ((exponent + 15) << 10) | (mantissa >> 13))};
    }
    if (exponent >= -24) {
        uint32_t sub = mantissa | 0x800000u;
        sub >>= (-14 - exponent);
        return llaisys::fp16_t{static_cast<uint16_t>(sign | (sub >> 13))};
    }
    return llaisys::fp16_t{sign};
}

__host__ __device__ inline float bf16ToFloat(llaisys::bf16_t value) {
    union {
        uint32_t bits;
        float value;
    } out{static_cast<uint32_t>(value._v) << 16};
    return out.value;
}

__host__ __device__ inline llaisys::bf16_t floatToBf16(float value) {
    union {
        uint32_t bits;
        float value;
    } in{0};
    in.value = value;
    const uint32_t rounding_bias = 0x7FFFu + ((in.bits >> 16) & 1u);
    return llaisys::bf16_t{static_cast<uint16_t>((in.bits + rounding_bias) >> 16)};
}

template <typename T>
__host__ __device__ inline float toFloat(T value) {
    return static_cast<float>(value);
}

template <>
__host__ __device__ inline float toFloat<llaisys::fp16_t>(llaisys::fp16_t value) {
    return fp16ToFloat(value);
}

template <>
__host__ __device__ inline float toFloat<llaisys::bf16_t>(llaisys::bf16_t value) {
    return bf16ToFloat(value);
}

template <typename T>
__host__ __device__ inline T fromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__host__ __device__ inline llaisys::fp16_t fromFloat<llaisys::fp16_t>(float value) {
    return floatToFp16(value);
}

template <>
__host__ __device__ inline llaisys::bf16_t fromFloat<llaisys::bf16_t>(float value) {
    return floatToBf16(value);
}

inline dim3 launch1D(size_t n, int block = 256) {
    return dim3(static_cast<unsigned int>((n + block - 1) / block));
}

} // namespace llaisys::device::nvidia
