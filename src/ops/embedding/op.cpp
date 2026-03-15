#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

#include <cstring>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        size_t row_bytes = weight->shape()[1] * weight->elementSize();
        const int64_t *idx = reinterpret_cast<const int64_t *>(index->data());
        const std::byte *w = weight->data();
        std::byte *o = out->data();
        for (size_t i = 0; i < index->shape()[0]; i++) {
            size_t row = static_cast<size_t>(idx[i]);
            const std::byte *src = w + row * row_bytes;
            std::byte *dst = o + i * row_bytes;
            std::memcpy(dst, src, row_bytes);
        }
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        {
            size_t row_bytes = weight->shape()[1] * weight->elementSize();
            const int64_t *idx = reinterpret_cast<const int64_t *>(index->data());
            const std::byte *w = weight->data();
            std::byte *o = out->data();
            for (size_t i = 0; i < index->shape()[0]; i++) {
                size_t row = static_cast<size_t>(idx[i]);
                const std::byte *src = w + row * row_bytes;
                std::byte *dst = o + i * row_bytes;
                std::memcpy(dst, src, row_bytes);
            }
            return;
        }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out, index, weight);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
