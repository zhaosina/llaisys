#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <limits>

namespace llaisys::ops {
template <typename T>
void argmax_cpu(int64_t *out_idx, T *out_val, const T *vals, size_t n) {
    if (n == 0) {
        *out_idx = 0;
        *out_val = utils::cast<T>(0.0f);
        return;
    }
    float max_v = utils::cast<float>(vals[0]);
    int64_t max_i = 0;
    for (size_t i = 1; i < n; i++) {
        float v = utils::cast<float>(vals[i]);
        if (v > max_v) {
            max_v = v;
            max_i = static_cast<int64_t>(i);
        }
    }
    *out_idx = max_i;
    *out_val = utils::cast<T>(max_v);
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            return argmax_cpu(reinterpret_cast<int64_t *>(max_idx->data()),
                              reinterpret_cast<float *>(max_val->data()),
                              reinterpret_cast<const float *>(vals->data()),
                              vals->numel());
        case LLAISYS_DTYPE_F16:
            return argmax_cpu(reinterpret_cast<int64_t *>(max_idx->data()),
                              reinterpret_cast<fp16_t *>(max_val->data()),
                              reinterpret_cast<const fp16_t *>(vals->data()),
                              vals->numel());
        case LLAISYS_DTYPE_BF16:
            return argmax_cpu(reinterpret_cast<int64_t *>(max_idx->data()),
                              reinterpret_cast<bf16_t *>(max_val->data()),
                              reinterpret_cast<const bf16_t *>(vals->data()),
                              vals->numel());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
        }
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            return argmax_cpu(reinterpret_cast<int64_t *>(max_idx->data()),
                              reinterpret_cast<float *>(max_val->data()),
                              reinterpret_cast<const float *>(vals->data()),
                              vals->numel());
        case LLAISYS_DTYPE_F16:
            return argmax_cpu(reinterpret_cast<int64_t *>(max_idx->data()),
                              reinterpret_cast<fp16_t *>(max_val->data()),
                              reinterpret_cast<const fp16_t *>(vals->data()),
                              vals->numel());
        case LLAISYS_DTYPE_BF16:
            return argmax_cpu(reinterpret_cast<int64_t *>(max_idx->data()),
                              reinterpret_cast<bf16_t *>(max_val->data()),
                              reinterpret_cast<const bf16_t *>(vals->data()),
                              vals->numel());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
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
