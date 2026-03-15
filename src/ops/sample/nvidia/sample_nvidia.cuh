#pragma once

#include "../../../tensor/tensor.hpp"

#include <cstdint>

namespace llaisys::ops::nvidia {
void sample(tensor_t out_idx, tensor_t vals, float temperature, int top_k, float top_p, uint64_t seed);
}
