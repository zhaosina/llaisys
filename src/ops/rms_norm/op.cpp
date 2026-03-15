#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif

#include <cmath>

namespace llaisys::ops {
template <typename T>
void rms_norm_cpu(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    for (size_t i = 0; i < rows; i++) {
        const T *row = in + i * cols;
        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float v = utils::cast<float>(row[j]);
            sum += v * v;
        }
        float mean = sum / static_cast<float>(cols);
        float inv = 1.0f / std::sqrt(mean + eps);
        T *out_row = out + i * cols;
        for (size_t j = 0; j < cols; j++) {
            float v = utils::cast<float>(row[j]);
            float w = utils::cast<float>(weight[j]);
            out_row[j] = utils::cast<T>(v * inv * w);
        }
    }
}

template <typename T>
void qwen_rms_norm_cpu(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    for (size_t i = 0; i < rows; i++) {
        const T *row = in + i * cols;
        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float v = utils::cast<float>(row[j]);
            sum += v * v;
        }
        float mean = sum / static_cast<float>(cols);
        float inv = 1.0f / std::sqrt(mean + eps);
        T *out_row = out + i * cols;
        for (size_t j = 0; j < cols; j++) {
            float v = utils::cast<float>(row[j]);
            T normalized = utils::cast<T>(v * inv);
            float w = utils::cast<float>(weight[j]);
            out_row[j] = utils::cast<T>(utils::cast<float>(normalized) * w);
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        size_t rows = in->shape()[0];
        size_t cols = in->shape()[1];
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rms_norm_cpu(reinterpret_cast<float *>(out->data()),
                                reinterpret_cast<const float *>(in->data()),
                                reinterpret_cast<const float *>(weight->data()),
                                rows, cols, eps);
        case LLAISYS_DTYPE_F16:
            return rms_norm_cpu(reinterpret_cast<fp16_t *>(out->data()),
                                reinterpret_cast<const fp16_t *>(in->data()),
                                reinterpret_cast<const fp16_t *>(weight->data()),
                                rows, cols, eps);
        case LLAISYS_DTYPE_BF16:
            return rms_norm_cpu(reinterpret_cast<bf16_t *>(out->data()),
                                reinterpret_cast<const bf16_t *>(in->data()),
                                reinterpret_cast<const bf16_t *>(weight->data()),
                                rows, cols, eps);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        {
            size_t rows = in->shape()[0];
            size_t cols = in->shape()[1];
            switch (out->dtype()) {
            case LLAISYS_DTYPE_F32:
                return rms_norm_cpu(reinterpret_cast<float *>(out->data()),
                                    reinterpret_cast<const float *>(in->data()),
                                    reinterpret_cast<const float *>(weight->data()),
                                    rows, cols, eps);
            case LLAISYS_DTYPE_F16:
                return rms_norm_cpu(reinterpret_cast<fp16_t *>(out->data()),
                                    reinterpret_cast<const fp16_t *>(in->data()),
                                    reinterpret_cast<const fp16_t *>(weight->data()),
                                    rows, cols, eps);
            case LLAISYS_DTYPE_BF16:
                return rms_norm_cpu(reinterpret_cast<bf16_t *>(out->data()),
                                    reinterpret_cast<const bf16_t *>(in->data()),
                                    reinterpret_cast<const bf16_t *>(weight->data()),
                                    rows, cols, eps);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
            }
        }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out, in, weight, eps);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void qwen_rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        size_t rows = in->shape()[0];
        size_t cols = in->shape()[1];
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return qwen_rms_norm_cpu(reinterpret_cast<float *>(out->data()),
                                     reinterpret_cast<const float *>(in->data()),
                                     reinterpret_cast<const float *>(weight->data()),
                                     rows, cols, eps);
        case LLAISYS_DTYPE_F16:
            return qwen_rms_norm_cpu(reinterpret_cast<fp16_t *>(out->data()),
                                     reinterpret_cast<const fp16_t *>(in->data()),
                                     reinterpret_cast<const fp16_t *>(weight->data()),
                                     rows, cols, eps);
        case LLAISYS_DTYPE_BF16:
            return qwen_rms_norm_cpu(reinterpret_cast<bf16_t *>(out->data()),
                                     reinterpret_cast<const bf16_t *>(in->data()),
                                     reinterpret_cast<const bf16_t *>(weight->data()),
                                     rows, cols, eps);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        {
            size_t rows = in->shape()[0];
            size_t cols = in->shape()[1];
            switch (out->dtype()) {
            case LLAISYS_DTYPE_F32:
                return qwen_rms_norm_cpu(reinterpret_cast<float *>(out->data()),
                                         reinterpret_cast<const float *>(in->data()),
                                         reinterpret_cast<const float *>(weight->data()),
                                         rows, cols, eps);
            case LLAISYS_DTYPE_F16:
                return qwen_rms_norm_cpu(reinterpret_cast<fp16_t *>(out->data()),
                                         reinterpret_cast<const fp16_t *>(in->data()),
                                         reinterpret_cast<const fp16_t *>(weight->data()),
                                         rows, cols, eps);
            case LLAISYS_DTYPE_BF16:
                return qwen_rms_norm_cpu(reinterpret_cast<bf16_t *>(out->data()),
                                         reinterpret_cast<const bf16_t *>(in->data()),
                                         reinterpret_cast<const bf16_t *>(weight->data()),
                                         rows, cols, eps);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
            }
        }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::qwen_rms_norm(out, in, weight, eps);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
}
