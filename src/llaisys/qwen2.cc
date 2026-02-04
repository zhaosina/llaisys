#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../core/llaisys_core.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace {
using llaisys::Tensor;
using llaisys::tensor_t;

struct Qwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    int device_id;
    LlaisysQwen2Weights weights;
    std::vector<llaisysTensor_t> owned_tensors;
    std::vector<llaisysTensor_t *> owned_arrays;
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    size_t cache_len;
};

llaisysTensor_t make_weight(Qwen2Model *model, const std::vector<size_t> &shape, bool zero = false) {
    auto t = Tensor::create(shape, model->meta.dtype, model->device, model->device_id);
    auto lt = new LlaisysTensor{t};
    model->owned_tensors.push_back(lt);
    if (zero && model->device == LLAISYS_DEVICE_CPU) {
        std::memset(t->data(), 0, t->numel() * t->elementSize());
    }
    return lt;
}

tensor_t make_tensor(Qwen2Model *model, const std::vector<size_t> &shape, llaisysDataType_t dtype) {
    return Tensor::create(shape, dtype, model->device, model->device_id);
}

tensor_t add_tensors(tensor_t a, tensor_t b) {
    auto out = Tensor::create(a->shape(), a->dtype(), a->deviceType(), a->deviceId());
    llaisys::ops::add(out, a, b);
    return out;
}

void copy_into_cache(tensor_t cache, tensor_t src, size_t start) {
    size_t row = src->shape()[1] * src->shape()[2];
    size_t bytes = src->numel() * src->elementSize();
    size_t offset = start * row * src->elementSize();
    std::memcpy(cache->data() + offset, src->data(), bytes);
}

int64_t infer_impl(Qwen2Model *model, const int64_t *token_ids, size_t ntoken) {
    if (ntoken == 0) {
        return model->meta.end_token;
    }
    if (ntoken < model->cache_len) {
        model->cache_len = 0;
    }
    size_t new_tokens = ntoken - model->cache_len;
    if (new_tokens == 0) {
        return model->meta.end_token;
    }
    CHECK_ARGUMENT(model->cache_len + new_tokens <= model->meta.maxseq, "sequence too long");

    auto idx = make_tensor(model, {new_tokens}, LLAISYS_DTYPE_I64);
    if (model->device == LLAISYS_DEVICE_CPU) {
        std::memcpy(idx->data(), token_ids + model->cache_len, new_tokens * sizeof(int64_t));
    } else {
        TO_BE_IMPLEMENTED();
    }

    auto hidden = make_tensor(model, {new_tokens, model->meta.hs}, model->meta.dtype);
    llaisys::ops::embedding(hidden, idx, model->weights.in_embed->tensor);

    for (size_t layer = 0; layer < model->meta.nlayer; layer++) {
        auto attn_norm = make_tensor(model, {new_tokens, model->meta.hs}, model->meta.dtype);
        llaisys::ops::rms_norm(attn_norm, hidden, model->weights.attn_norm_w[layer]->tensor, model->meta.epsilon);

        size_t q_dim = model->meta.nh * model->meta.dh;
        size_t kv_dim = model->meta.nkvh * model->meta.dh;

        auto q2d = make_tensor(model, {new_tokens, q_dim}, model->meta.dtype);
        auto k2d = make_tensor(model, {new_tokens, kv_dim}, model->meta.dtype);
        auto v2d = make_tensor(model, {new_tokens, kv_dim}, model->meta.dtype);
        llaisys::ops::linear(q2d, attn_norm, model->weights.attn_q_w[layer]->tensor,
                             model->weights.attn_q_b[layer] ? model->weights.attn_q_b[layer]->tensor : nullptr);
        llaisys::ops::linear(k2d, attn_norm, model->weights.attn_k_w[layer]->tensor,
                             model->weights.attn_k_b[layer] ? model->weights.attn_k_b[layer]->tensor : nullptr);
        llaisys::ops::linear(v2d, attn_norm, model->weights.attn_v_w[layer]->tensor,
                             model->weights.attn_v_b[layer] ? model->weights.attn_v_b[layer]->tensor : nullptr);

        auto q3 = q2d->view({new_tokens, model->meta.nh, model->meta.dh});
        auto k3 = k2d->view({new_tokens, model->meta.nkvh, model->meta.dh});
        auto v3 = v2d->view({new_tokens, model->meta.nkvh, model->meta.dh});

        auto pos_ids = make_tensor(model, {new_tokens}, LLAISYS_DTYPE_I64);
        if (model->device == LLAISYS_DEVICE_CPU) {
            auto p = reinterpret_cast<int64_t *>(pos_ids->data());
            for (size_t i = 0; i < new_tokens; i++) {
                p[i] = static_cast<int64_t>(model->cache_len + i);
            }
        } else {
            TO_BE_IMPLEMENTED();
        }

        auto q_rope = make_tensor(model, {new_tokens, model->meta.nh, model->meta.dh}, model->meta.dtype);
        auto k_rope = make_tensor(model, {new_tokens, model->meta.nkvh, model->meta.dh}, model->meta.dtype);
        llaisys::ops::rope(q_rope, q3, pos_ids, model->meta.theta);
        llaisys::ops::rope(k_rope, k3, pos_ids, model->meta.theta);

        copy_into_cache(model->k_cache[layer], k_rope, model->cache_len);
        copy_into_cache(model->v_cache[layer], v3, model->cache_len);

        size_t total_len = model->cache_len + new_tokens;
        auto k_cache = model->k_cache[layer]->slice(0, 0, total_len);
        auto v_cache = model->v_cache[layer]->slice(0, 0, total_len);

        auto attn_val = make_tensor(model, {new_tokens, model->meta.nh, model->meta.dh}, model->meta.dtype);
        float scale = 1.0f / std::sqrt(static_cast<float>(model->meta.dh));
        llaisys::ops::self_attention(attn_val, q_rope, k_cache, v_cache, scale);

        auto attn_2d = attn_val->view({new_tokens, model->meta.hs});
        auto proj = make_tensor(model, {new_tokens, model->meta.hs}, model->meta.dtype);
        llaisys::ops::linear(proj, attn_2d, model->weights.attn_o_w[layer]->tensor, nullptr);
        hidden = add_tensors(hidden, proj);

        auto mlp_norm = make_tensor(model, {new_tokens, model->meta.hs}, model->meta.dtype);
        llaisys::ops::rms_norm(mlp_norm, hidden, model->weights.mlp_norm_w[layer]->tensor, model->meta.epsilon);

        auto gate = make_tensor(model, {new_tokens, model->meta.di}, model->meta.dtype);
        auto up = make_tensor(model, {new_tokens, model->meta.di}, model->meta.dtype);
        llaisys::ops::linear(gate, mlp_norm, model->weights.mlp_gate_w[layer]->tensor, nullptr);
        llaisys::ops::linear(up, mlp_norm, model->weights.mlp_up_w[layer]->tensor, nullptr);

        auto swiglu_out = make_tensor(model, {new_tokens, model->meta.di}, model->meta.dtype);
        llaisys::ops::swiglu(swiglu_out, gate, up);

        auto down = make_tensor(model, {new_tokens, model->meta.hs}, model->meta.dtype);
        llaisys::ops::linear(down, swiglu_out, model->weights.mlp_down_w[layer]->tensor, nullptr);
        hidden = add_tensors(hidden, down);
    }

    auto final_norm = make_tensor(model, {new_tokens, model->meta.hs}, model->meta.dtype);
    llaisys::ops::rms_norm(final_norm, hidden, model->weights.out_norm_w->tensor, model->meta.epsilon);
    auto last_hidden = final_norm->slice(0, new_tokens - 1, new_tokens);
    auto logits = make_tensor(model, {1, model->meta.voc}, model->meta.dtype);
    llaisys::ops::linear(logits, last_hidden, model->weights.out_embed->tensor, nullptr);

    auto logits_view = logits->view({model->meta.voc});
    auto max_idx = make_tensor(model, {1}, LLAISYS_DTYPE_I64);
    auto max_val = make_tensor(model, {1}, model->meta.dtype);
    llaisys::ops::argmax(max_idx, max_val, logits_view);

    int64_t next_id = 0;
    if (model->device == LLAISYS_DEVICE_CPU) {
        next_id = *reinterpret_cast<int64_t *>(max_idx->data());
    } else {
        TO_BE_IMPLEMENTED();
    }

    model->cache_len += new_tokens;
    return next_id;
}
} // namespace

__C {
    struct LlaisysQwen2Model {
        Qwen2Model *impl;
    };

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        auto model = new Qwen2Model{};
        model->meta = *meta;
        model->device = device;
        model->device_id = (device_ids && ndevice > 0) ? device_ids[0] : 0;
        model->cache_len = 0;

        model->weights.in_embed = make_weight(model, {meta->voc, meta->hs});
        model->weights.out_embed = make_weight(model, {meta->voc, meta->hs});
        model->weights.out_norm_w = make_weight(model, {meta->hs});

        model->weights.attn_norm_w = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_q_w = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_q_b = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_k_w = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_k_b = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_v_w = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_v_b = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_o_w = new llaisysTensor_t[meta->nlayer];
        model->weights.mlp_norm_w = new llaisysTensor_t[meta->nlayer];
        model->weights.mlp_gate_w = new llaisysTensor_t[meta->nlayer];
        model->weights.mlp_up_w = new llaisysTensor_t[meta->nlayer];
        model->weights.mlp_down_w = new llaisysTensor_t[meta->nlayer];

        model->owned_arrays.push_back(model->weights.attn_norm_w);
        model->owned_arrays.push_back(model->weights.attn_q_w);
        model->owned_arrays.push_back(model->weights.attn_q_b);
        model->owned_arrays.push_back(model->weights.attn_k_w);
        model->owned_arrays.push_back(model->weights.attn_k_b);
        model->owned_arrays.push_back(model->weights.attn_v_w);
        model->owned_arrays.push_back(model->weights.attn_v_b);
        model->owned_arrays.push_back(model->weights.attn_o_w);
        model->owned_arrays.push_back(model->weights.mlp_norm_w);
        model->owned_arrays.push_back(model->weights.mlp_gate_w);
        model->owned_arrays.push_back(model->weights.mlp_up_w);
        model->owned_arrays.push_back(model->weights.mlp_down_w);

        for (size_t i = 0; i < meta->nlayer; i++) {
            model->weights.attn_norm_w[i] = make_weight(model, {meta->hs});
            model->weights.attn_q_w[i] = make_weight(model, {meta->nh * meta->dh, meta->hs});
            model->weights.attn_q_b[i] = make_weight(model, {meta->nh * meta->dh}, true);
            model->weights.attn_k_w[i] = make_weight(model, {meta->nkvh * meta->dh, meta->hs});
            model->weights.attn_k_b[i] = make_weight(model, {meta->nkvh * meta->dh}, true);
            model->weights.attn_v_w[i] = make_weight(model, {meta->nkvh * meta->dh, meta->hs});
            model->weights.attn_v_b[i] = make_weight(model, {meta->nkvh * meta->dh}, true);
            model->weights.attn_o_w[i] = make_weight(model, {meta->hs, meta->nh * meta->dh});
            model->weights.mlp_norm_w[i] = make_weight(model, {meta->hs});
            model->weights.mlp_gate_w[i] = make_weight(model, {meta->di, meta->hs});
            model->weights.mlp_up_w[i] = make_weight(model, {meta->di, meta->hs});
            model->weights.mlp_down_w[i] = make_weight(model, {meta->hs, meta->di});
        }

        model->k_cache.resize(meta->nlayer);
        model->v_cache.resize(meta->nlayer);
        for (size_t i = 0; i < meta->nlayer; i++) {
            model->k_cache[i] = Tensor::create({meta->maxseq, meta->nkvh, meta->dh}, meta->dtype, device, model->device_id);
            model->v_cache[i] = Tensor::create({meta->maxseq, meta->nkvh, meta->dh}, meta->dtype, device, model->device_id);
        }

        auto wrapper = new LlaisysQwen2Model{};
        wrapper->impl = model;
        return wrapper;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        if (!model) {
            return;
        }
        auto impl = model->impl;
        if (impl) {
            for (auto t : impl->owned_tensors) {
                delete t;
            }
            for (auto arr : impl->owned_arrays) {
                delete[] arr;
            }
            delete impl;
        }
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        return model ? &model->impl->weights : nullptr;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        if (model == nullptr || model->impl == nullptr) {
            return -1;
        }
        try {
            return infer_impl(model->impl, token_ids, ntoken);
        } catch (...) {
            return -1;
        }
    }
}
