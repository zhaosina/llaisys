#include "self_attention_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <cmath>
#include <limits>
#include <vector>

namespace {
template <typename T>
void selfAttentionHost(T *out,
                       const T *q,
                       const T *k,
                       const T *v,
                       size_t qlen,
                       size_t kvlen,
                       size_t nhead,
                       size_t nkvh,
                       size_t d,
                       size_t dv,
                       float scale) {
    const size_t repeat = nhead / nkvh;
    std::vector<float> scores(kvlen);
    const int offset = static_cast<int>(kvlen) - static_cast<int>(qlen);
    for (size_t h = 0; h < nhead; h++) {
        const size_t hk = h / repeat;
        for (size_t i = 0; i < qlen; i++) {
            const int max_j = static_cast<int>(i) + offset;
            float max_score = -std::numeric_limits<float>::infinity();
            bool any = false;
            const size_t q_base = (i * nhead + h) * d;
            for (size_t j = 0; j < kvlen; j++) {
                if (static_cast<int>(j) > max_j) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const size_t k_base = (j * nkvh + hk) * d;
                float dot = 0.0f;
                for (size_t t = 0; t < d; t++) {
                    dot += llaisys::device::nvidia::toFloat(q[q_base + t]) *
                           llaisys::device::nvidia::toFloat(k[k_base + t]);
                }
                const float score = dot * scale;
                scores[j] = score;
                if (!any || score > max_score) {
                    max_score = score;
                    any = true;
                }
            }
            const size_t out_base = (i * nhead + h) * dv;
            if (!any) {
                for (size_t t = 0; t < dv; t++) {
                    out[out_base + t] = llaisys::device::nvidia::fromFloat<T>(0.0f);
                }
                continue;
            }
            float sum = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                if (static_cast<int>(j) > max_j) {
                    continue;
                }
                const float weight = std::exp(scores[j] - max_score);
                scores[j] = weight;
                sum += weight;
            }
            const float inv_sum = 1.0f / sum;
            for (size_t t = 0; t < dv; t++) {
                float acc = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    if (static_cast<int>(j) > max_j) {
                        continue;
                    }
                    const size_t v_base = (j * nkvh + hk) * dv;
                    acc += (scores[j] * inv_sum) * llaisys::device::nvidia::toFloat(v[v_base + t]);
                }
                out[out_base + t] = llaisys::device::nvidia::fromFloat<T>(acc);
            }
        }
    }
}

template <typename T>
void launchSelfAttention(llaisys::tensor_t attn_val, llaisys::tensor_t q, llaisys::tensor_t k, llaisys::tensor_t v, float scale) {
    std::vector<T> host_q(q->numel());
    std::vector<T> host_k(k->numel());
    std::vector<T> host_v(v->numel());
    std::vector<T> host_out(attn_val->numel());

    CUDA_CHECK(cudaMemcpy(host_q.data(), q->data(), q->numel() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_k.data(), k->data(), k->numel() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_v.data(), v->data(), v->numel() * sizeof(T), cudaMemcpyDeviceToHost));

    selfAttentionHost(
        host_out.data(),
        host_q.data(),
        host_k.data(),
        host_v.data(),
        q->shape()[0],
        k->shape()[0],
        q->shape()[1],
        k->shape()[1],
        q->shape()[2],
        v->shape()[2],
        scale);

    CUDA_CHECK(cudaMemcpy(attn_val->data(), host_out.data(), host_out.size() * sizeof(T), cudaMemcpyHostToDevice));
}
} // namespace

namespace llaisys::ops::nvidia {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    switch (attn_val->dtype()) {
    case LLAISYS_DTYPE_F32:
        return launchSelfAttention<float>(attn_val, q, k, v, scale);
    case LLAISYS_DTYPE_F16:
        return launchSelfAttention<fp16_t>(attn_val, q, k, v, scale);
    case LLAISYS_DTYPE_BF16:
        return launchSelfAttention<bf16_t>(attn_val, q, k, v, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
}
} // namespace llaisys::ops::nvidia
