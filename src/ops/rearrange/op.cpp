#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <vector>

namespace llaisys::ops {
template <typename T>
void rearrange_cpu(T *out,
                   const T *in,
                   const std::vector<size_t> &shape,
                   const std::vector<ptrdiff_t> &out_strides,
                   const std::vector<ptrdiff_t> &in_strides) {
    size_t ndim = shape.size();
    size_t numel = 1;
    for (size_t s : shape) {
        numel *= s;
    }
    std::vector<size_t> idx(ndim, 0);
    for (size_t n = 0; n < numel; n++) {
        size_t out_offset = 0;
        size_t in_offset = 0;
        for (size_t d = 0; d < ndim; d++) {
            out_offset += idx[d] * static_cast<size_t>(out_strides[d]);
            in_offset += idx[d] * static_cast<size_t>(in_strides[d]);
        }
        out[out_offset] = in[in_offset];
        for (size_t d = ndim; d-- > 0;) {
            idx[d]++;
            if (idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
}

void rearrange(tensor_t out, tensor_t in) {
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rearrange_cpu(reinterpret_cast<float *>(out->data()),
                                 reinterpret_cast<const float *>(in->data()),
                                 out->shape(), out->strides(), in->strides());
        case LLAISYS_DTYPE_F16:
            return rearrange_cpu(reinterpret_cast<fp16_t *>(out->data()),
                                 reinterpret_cast<const fp16_t *>(in->data()),
                                 out->shape(), out->strides(), in->strides());
        case LLAISYS_DTYPE_BF16:
            return rearrange_cpu(reinterpret_cast<bf16_t *>(out->data()),
                                 reinterpret_cast<const bf16_t *>(in->data()),
                                 out->shape(), out->strides(), in->strides());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rearrange_cpu(reinterpret_cast<float *>(out->data()),
                                 reinterpret_cast<const float *>(in->data()),
                                 out->shape(), out->strides(), in->strides());
        case LLAISYS_DTYPE_F16:
            return rearrange_cpu(reinterpret_cast<fp16_t *>(out->data()),
                                 reinterpret_cast<const fp16_t *>(in->data()),
                                 out->shape(), out->strides(), in->strides());
        case LLAISYS_DTYPE_BF16:
            return rearrange_cpu(reinterpret_cast<bf16_t *>(out->data()),
                                 reinterpret_cast<const bf16_t *>(in->data()),
                                 out->shape(), out->strides(), in->strides());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
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
}
