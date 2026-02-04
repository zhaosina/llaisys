#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops {
template <typename T>
void self_attention_cpu(T *out,
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
    size_t repeat = nhead / nkvh;
    std::vector<float> scores(kvlen);
    int offset = static_cast<int>(kvlen) - static_cast<int>(qlen);
    for (size_t h = 0; h < nhead; h++) {
        size_t hk = h / repeat;
        for (size_t i = 0; i < qlen; i++) {
            int max_j = static_cast<int>(i) + offset;
            float max_score = -std::numeric_limits<float>::infinity();
            bool any = false;
            size_t q_base = (i * nhead + h) * d;
            for (size_t j = 0; j < kvlen; j++) {
                if (static_cast<int>(j) > max_j) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                size_t k_base = (j * nkvh + hk) * d;
                float dot = 0.0f;
                for (size_t t = 0; t < d; t++) {
                    dot += utils::cast<float>(q[q_base + t]) * utils::cast<float>(k[k_base + t]);
                }
                float s = dot * scale;
                scores[j] = s;
                if (!any || s > max_score) {
                    max_score = s;
                    any = true;
                }
            }
            if (!any) {
                size_t out_base = (i * nhead + h) * dv;
                for (size_t t = 0; t < dv; t++) {
                    out[out_base + t] = utils::cast<T>(0.0f);
                }
                continue;
            }
            float sum = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                if (static_cast<int>(j) > max_j) {
                    continue;
                }
                float w = std::exp(scores[j] - max_score);
                scores[j] = w;
                sum += w;
            }
            float inv_sum = 1.0f / sum;
            size_t out_base = (i * nhead + h) * dv;
            for (size_t t = 0; t < dv; t++) {
                float acc = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    if (static_cast<int>(j) > max_j) {
                        continue;
                    }
                    size_t v_base = (j * nkvh + hk) * dv;
                    acc += (scores[j] * inv_sum) * utils::cast<float>(v[v_base + t]);
                }
                out[out_base + t] = utils::cast<T>(acc);
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        size_t qlen = q->shape()[0];
        size_t kvlen = k->shape()[0];
        size_t nhead = q->shape()[1];
        size_t nkvh = k->shape()[1];
        size_t d = q->shape()[2];
        size_t dv = v->shape()[2];
        switch (attn_val->dtype()) {
        case LLAISYS_DTYPE_F32:
            return self_attention_cpu(reinterpret_cast<float *>(attn_val->data()),
                                      reinterpret_cast<const float *>(q->data()),
                                      reinterpret_cast<const float *>(k->data()),
                                      reinterpret_cast<const float *>(v->data()),
                                      qlen, kvlen, nhead, nkvh, d, dv, scale);
        case LLAISYS_DTYPE_F16:
            return self_attention_cpu(reinterpret_cast<fp16_t *>(attn_val->data()),
                                      reinterpret_cast<const fp16_t *>(q->data()),
                                      reinterpret_cast<const fp16_t *>(k->data()),
                                      reinterpret_cast<const fp16_t *>(v->data()),
                                      qlen, kvlen, nhead, nkvh, d, dv, scale);
        case LLAISYS_DTYPE_BF16:
            return self_attention_cpu(reinterpret_cast<bf16_t *>(attn_val->data()),
                                      reinterpret_cast<const bf16_t *>(q->data()),
                                      reinterpret_cast<const bf16_t *>(k->data()),
                                      reinterpret_cast<const bf16_t *>(v->data()),
                                      qlen, kvlen, nhead, nkvh, d, dv, scale);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
        }
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        {
            size_t qlen = q->shape()[0];
            size_t kvlen = k->shape()[0];
            size_t nhead = q->shape()[1];
            size_t nkvh = k->shape()[1];
            size_t d = q->shape()[2];
            size_t dv = v->shape()[2];
            switch (attn_val->dtype()) {
            case LLAISYS_DTYPE_F32:
                return self_attention_cpu(reinterpret_cast<float *>(attn_val->data()),
                                          reinterpret_cast<const float *>(q->data()),
                                          reinterpret_cast<const float *>(k->data()),
                                          reinterpret_cast<const float *>(v->data()),
                                          qlen, kvlen, nhead, nkvh, d, dv, scale);
            case LLAISYS_DTYPE_F16:
                return self_attention_cpu(reinterpret_cast<fp16_t *>(attn_val->data()),
                                          reinterpret_cast<const fp16_t *>(q->data()),
                                          reinterpret_cast<const fp16_t *>(k->data()),
                                          reinterpret_cast<const fp16_t *>(v->data()),
                                          qlen, kvlen, nhead, nkvh, d, dv, scale);
            case LLAISYS_DTYPE_BF16:
                return self_attention_cpu(reinterpret_cast<bf16_t *>(attn_val->data()),
                                          reinterpret_cast<const bf16_t *>(q->data()),
                                          reinterpret_cast<const bf16_t *>(k->data()),
                                          reinterpret_cast<const bf16_t *>(v->data()),
                                          qlen, kvlen, nhead, nkvh, d, dv, scale);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
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
}
