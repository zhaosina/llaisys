#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    CHECK_ARGUMENT(out->ndim() == 2, "Linear: output tensor must be 2D.");
    CHECK_ARGUMENT(in->ndim() == 2, "Linear: input tensor must be 2D.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Linear: weight tensor must be 2D.");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "Linear: output rows must match input rows.");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0], "Linear: output cols must match weight rows.");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "Linear: input cols must match weight cols.");
    if (bias) {
        CHECK_ARGUMENT(bias->ndim() == 1, "Linear: bias tensor must be 1D.");
        CHECK_ARGUMENT(bias->shape()[0] == weight->shape()[0], "Linear: bias length must match output cols.");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out, in, weight, bias);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out, in, weight, bias);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out, in, weight, bias);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
