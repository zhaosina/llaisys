#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

#include <cmath>
#include <vector>

namespace llaisys::ops {
template <typename T>
void rope_cpu(T *out, const T *in, const int64_t *pos, size_t seqlen, size_t nhead, size_t d, float theta) {
    size_t half = d / 2;
    std::vector<float> denom(half);
    for (size_t j = 0; j < half; j++) {
        float expv = (2.0f * static_cast<float>(j)) / static_cast<float>(d);
        denom[j] = std::pow(theta, expv);
    }
    for (size_t i = 0; i < seqlen; i++) {
        float pos_f = static_cast<float>(pos[i]);
        for (size_t h = 0; h < nhead; h++) {
            size_t base = (i * nhead + h) * d;
            for (size_t j = 0; j < half; j++) {
                float angle = pos_f / denom[j];
                float s = std::sin(angle);
                float c = std::cos(angle);
                float a = utils::cast<float>(in[base + j]);
                float b = utils::cast<float>(in[base + j + half]);
                out[base + j] = utils::cast<T>(a * c - b * s);
                out[base + j + half] = utils::cast<T>(b * c + a * s);
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        size_t seqlen = out->shape()[0];
        size_t nhead = out->shape()[1];
        size_t d = out->shape()[2];
        const int64_t *pos = reinterpret_cast<const int64_t *>(pos_ids->data());
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rope_cpu(reinterpret_cast<float *>(out->data()),
                            reinterpret_cast<const float *>(in->data()),
                            pos, seqlen, nhead, d, theta);
        case LLAISYS_DTYPE_F16:
            return rope_cpu(reinterpret_cast<fp16_t *>(out->data()),
                            reinterpret_cast<const fp16_t *>(in->data()),
                            pos, seqlen, nhead, d, theta);
        case LLAISYS_DTYPE_BF16:
            return rope_cpu(reinterpret_cast<bf16_t *>(out->data()),
                            reinterpret_cast<const bf16_t *>(in->data()),
                            pos, seqlen, nhead, d, theta);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        {
            size_t seqlen = out->shape()[0];
            size_t nhead = out->shape()[1];
            size_t d = out->shape()[2];
            const int64_t *pos = reinterpret_cast<const int64_t *>(pos_ids->data());
            switch (out->dtype()) {
            case LLAISYS_DTYPE_F32:
                return rope_cpu(reinterpret_cast<float *>(out->data()),
                                reinterpret_cast<const float *>(in->data()),
                                pos, seqlen, nhead, d, theta);
            case LLAISYS_DTYPE_F16:
                return rope_cpu(reinterpret_cast<fp16_t *>(out->data()),
                                reinterpret_cast<const fp16_t *>(in->data()),
                                pos, seqlen, nhead, d, theta);
            case LLAISYS_DTYPE_BF16:
                return rope_cpu(reinterpret_cast<bf16_t *>(out->data()),
                                reinterpret_cast<const bf16_t *>(in->data()),
                                pos, seqlen, nhead, d, theta);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
            }
        }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out, in, pos_ids, theta);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
}
