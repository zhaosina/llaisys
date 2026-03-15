#include "sample_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace {
template <typename T>
int64_t sample_host(const T *vals, size_t n, float temperature, int top_k, float top_p, uint64_t seed) {
    if (temperature <= 0.0f || top_k == 1) {
        float max_v = llaisys::utils::cast<float>(vals[0]);
        int64_t max_i = 0;
        for (size_t i = 1; i < n; i++) {
            const float v = llaisys::utils::cast<float>(vals[i]);
            if (v > max_v) {
                max_v = v;
                max_i = static_cast<int64_t>(i);
            }
        }
        return max_i;
    }

    struct Candidate {
        int64_t index;
        float scaled_logit;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(n);
    const float inv_temperature = 1.0f / temperature;
    for (size_t i = 0; i < n; i++) {
        candidates.push_back({static_cast<int64_t>(i), llaisys::utils::cast<float>(vals[i]) * inv_temperature});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
        return a.scaled_logit > b.scaled_logit;
    });

    size_t keep = candidates.size();
    if (top_k > 0 && static_cast<size_t>(top_k) < keep) {
        keep = static_cast<size_t>(top_k);
    }
    candidates.resize(keep);

    const float max_logit = candidates.front().scaled_logit;
    std::vector<float> probs(candidates.size());
    float denom = 0.0f;
    for (size_t i = 0; i < candidates.size(); i++) {
        probs[i] = std::exp(candidates[i].scaled_logit - max_logit);
        denom += probs[i];
    }
    for (float &p : probs) {
        p /= denom;
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        float cumulative = 0.0f;
        size_t nucleus = 0;
        for (; nucleus < probs.size(); nucleus++) {
            cumulative += probs[nucleus];
            if (cumulative >= top_p) {
                nucleus++;
                break;
            }
        }
        nucleus = std::max<size_t>(1, std::min(nucleus, probs.size()));
        candidates.resize(nucleus);
        probs.resize(nucleus);
        const float kept_sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (float &p : probs) {
            p /= kept_sum;
        }
    }

    std::mt19937_64 rng(seed);
    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    return candidates[dist(rng)].index;
}

template <typename T>
void sample_device(llaisys::tensor_t out_idx, llaisys::tensor_t vals, float temperature, int top_k, float top_p, uint64_t seed) {
    std::vector<T> host_vals(vals->numel());
    CUDA_CHECK(cudaMemcpy(host_vals.data(), vals->data(), vals->numel() * sizeof(T), cudaMemcpyDeviceToHost));
    int64_t sampled = sample_host(host_vals.data(), host_vals.size(), temperature, top_k, top_p, seed);
    CUDA_CHECK(cudaMemcpy(out_idx->data(), &sampled, sizeof(sampled), cudaMemcpyHostToDevice));
}
} // namespace

namespace llaisys::ops::nvidia {
void sample(tensor_t out_idx, tensor_t vals, float temperature, int top_k, float top_p, uint64_t seed) {
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        return sample_device<float>(out_idx, vals, temperature, top_k, top_p, seed);
    case LLAISYS_DTYPE_F16:
        return sample_device<fp16_t>(out_idx, vals, temperature, top_k, top_p, seed);
    case LLAISYS_DTYPE_BF16:
        return sample_device<bf16_t>(out_idx, vals, temperature, top_k, top_p, seed);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}
} // namespace llaisys::ops::nvidia
