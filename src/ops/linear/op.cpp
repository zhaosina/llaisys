#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>

namespace llaisys::ops {
template <typename T>
void linear_cpu(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = 0.0f;
            if (bias != nullptr) {
                acc = utils::cast<float>(bias[j]);
            }
            const T *in_row = in + i * k;
            const T *w_row = weight + j * k;
            for (size_t t = 0; t < k; t++) {
                acc += utils::cast<float>(in_row[t]) * utils::cast<float>(w_row[t]);
            }
            out[i * n + j] = utils::cast<T>(acc);
        }
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        size_t m = out->shape()[0];
        size_t n = out->shape()[1];
        size_t k = in->shape()[1];
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return linear_cpu(reinterpret_cast<float *>(out->data()),
                              reinterpret_cast<const float *>(in->data()),
                              reinterpret_cast<const float *>(weight->data()),
                              bias ? reinterpret_cast<const float *>(bias->data()) : nullptr,
                              m, n, k);
        case LLAISYS_DTYPE_F16:
            return linear_cpu(reinterpret_cast<fp16_t *>(out->data()),
                              reinterpret_cast<const fp16_t *>(in->data()),
                              reinterpret_cast<const fp16_t *>(weight->data()),
                              bias ? reinterpret_cast<const fp16_t *>(bias->data()) : nullptr,
                              m, n, k);
        case LLAISYS_DTYPE_BF16:
            return linear_cpu(reinterpret_cast<bf16_t *>(out->data()),
                              reinterpret_cast<const bf16_t *>(in->data()),
                              reinterpret_cast<const bf16_t *>(weight->data()),
                              bias ? reinterpret_cast<const bf16_t *>(bias->data()) : nullptr,
                              m, n, k);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        {
            size_t m = out->shape()[0];
            size_t n = out->shape()[1];
            size_t k = in->shape()[1];
            switch (out->dtype()) {
            case LLAISYS_DTYPE_F32:
                return linear_cpu(reinterpret_cast<float *>(out->data()),
                                  reinterpret_cast<const float *>(in->data()),
                                  reinterpret_cast<const float *>(weight->data()),
                                  bias ? reinterpret_cast<const float *>(bias->data()) : nullptr,
                                  m, n, k);
            case LLAISYS_DTYPE_F16:
                return linear_cpu(reinterpret_cast<fp16_t *>(out->data()),
                                  reinterpret_cast<const fp16_t *>(in->data()),
                                  reinterpret_cast<const fp16_t *>(weight->data()),
                                  bias ? reinterpret_cast<const fp16_t *>(bias->data()) : nullptr,
                                  m, n, k);
            case LLAISYS_DTYPE_BF16:
                return linear_cpu(reinterpret_cast<bf16_t *>(out->data()),
                                  reinterpret_cast<const bf16_t *>(in->data()),
                                  reinterpret_cast<const bf16_t *>(weight->data()),
                                  bias ? reinterpret_cast<const bf16_t *>(bias->data()) : nullptr,
                                  m, n, k);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
            }
        }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
