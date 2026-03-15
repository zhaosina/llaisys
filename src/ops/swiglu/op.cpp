#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"
#endif

#include <cmath>

namespace llaisys::ops {
template <typename T>
void swiglu_cpu(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float g = utils::cast<float>(gate[i]);
        float u = utils::cast<float>(up[i]);
        float val = u * (g / (1.0f + std::exp(-g)));
        out[i] = utils::cast<T>(val);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return swiglu_cpu(reinterpret_cast<float *>(out->data()),
                              reinterpret_cast<const float *>(gate->data()),
                              reinterpret_cast<const float *>(up->data()),
                              out->numel());
        case LLAISYS_DTYPE_F16:
            return swiglu_cpu(reinterpret_cast<fp16_t *>(out->data()),
                              reinterpret_cast<const fp16_t *>(gate->data()),
                              reinterpret_cast<const fp16_t *>(up->data()),
                              out->numel());
        case LLAISYS_DTYPE_BF16:
            return swiglu_cpu(reinterpret_cast<bf16_t *>(out->data()),
                              reinterpret_cast<const bf16_t *>(gate->data()),
                              reinterpret_cast<const bf16_t *>(up->data()),
                              out->numel());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return swiglu_cpu(reinterpret_cast<float *>(out->data()),
                              reinterpret_cast<const float *>(gate->data()),
                              reinterpret_cast<const float *>(up->data()),
                              out->numel());
        case LLAISYS_DTYPE_F16:
            return swiglu_cpu(reinterpret_cast<fp16_t *>(out->data()),
                              reinterpret_cast<const fp16_t *>(gate->data()),
                              reinterpret_cast<const fp16_t *>(up->data()),
                              out->numel());
        case LLAISYS_DTYPE_BF16:
            return swiglu_cpu(reinterpret_cast<bf16_t *>(out->data()),
                              reinterpret_cast<const bf16_t *>(gate->data()),
                              reinterpret_cast<const bf16_t *>(up->data()),
                              out->numel());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(out, gate, up);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
}
