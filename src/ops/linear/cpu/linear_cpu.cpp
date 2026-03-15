#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <vector>

#ifdef LLAISYS_CPU_AVX2
#include <immintrin.h>
#endif

#ifdef LLAISYS_USE_OPENBLAS
#include <cblas.h>
#endif

#include "linear_cpu.hpp"

#include "../../../utils.hpp"

namespace {
constexpr size_t kOmpGemvThreshold = 1ull << 18;
constexpr size_t kOmpGemmThreshold = 1ull << 20;
constexpr size_t kMTile = 4;
constexpr size_t kNTile = 64;
constexpr size_t kKTile = 256;

bool use_legacy_linear() {
    static const bool enabled = []() {
        const char *env = std::getenv("LLAISYS_LINEAR_USE_LEGACY");
        if (env == nullptr) {
            return false;
        }
        return env[0] == '1' || env[0] == 'y' || env[0] == 'Y' || env[0] == 't' || env[0] == 'T';
    }();
    return enabled;
}

inline size_t ceil_div(size_t value, size_t divisor) {
    return (value + divisor - 1) / divisor;
}

template <typename T>
void linear_reference(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = bias ? llaisys::utils::cast<float>(bias[j]) : 0.0f;
            const T *in_row = in + i * k;
            const T *w_row = weight + j * k;
            for (size_t t = 0; t < k; t++) {
                acc += llaisys::utils::cast<float>(in_row[t]) * llaisys::utils::cast<float>(w_row[t]);
            }
            out[i * n + j] = llaisys::utils::cast<T>(acc);
        }
    }
}

#ifdef LLAISYS_CPU_AVX2
inline float hsum256(__m256 value) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, value);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}
#endif

inline float dot_product_f32(const float *lhs, const float *rhs, size_t len) {
#ifdef LLAISYS_CPU_AVX2
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t t = 0;
    for (; t + 31 < len; t += 32) {
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(lhs + t), _mm256_loadu_ps(rhs + t)));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(lhs + t + 8), _mm256_loadu_ps(rhs + t + 8)));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_loadu_ps(lhs + t + 16), _mm256_loadu_ps(rhs + t + 16)));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_loadu_ps(lhs + t + 24), _mm256_loadu_ps(rhs + t + 24)));
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; t + 7 < len; t += 8) {
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(lhs + t), _mm256_loadu_ps(rhs + t)));
    }

    float sum = hsum256(acc);
    for (; t < len; t++) {
        sum += lhs[t] * rhs[t];
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t t = 0; t < len; t++) {
        sum += lhs[t] * rhs[t];
    }
    return sum;
#endif
}

template <typename T>
void convert_to_float(float *dst, const T *src, size_t len) {
    for (size_t i = 0; i < len; i++) {
        dst[i] = llaisys::utils::cast<float>(src[i]);
    }
}

void linear_f32_gemv(float *out, const float *in_row, const float *weight, const float *bias, size_t n, size_t k) {
#ifdef _OPENMP
    if (n * k >= kOmpGemvThreshold) {
#pragma omp parallel for schedule(static)
        for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(n); j++) {
            float acc = bias ? bias[j] : 0.0f;
            acc += dot_product_f32(in_row, weight + static_cast<size_t>(j) * k, k);
            out[j] = acc;
        }
        return;
    }
#endif

    for (size_t j = 0; j < n; j++) {
        float acc = bias ? bias[j] : 0.0f;
        acc += dot_product_f32(in_row, weight + j * k, k);
        out[j] = acc;
    }
}

void linear_f32_gemm(float *out, const float *in, const float *weight, const float *bias, size_t m, size_t n, size_t k) {
    const size_t mt = ceil_div(m, kMTile);
    const size_t nt = ceil_div(n, kNTile);

#ifdef _OPENMP
    if (m * n * k >= kOmpGemmThreshold) {
#pragma omp parallel for collapse(2) schedule(static)
        for (ptrdiff_t ib = 0; ib < static_cast<ptrdiff_t>(mt); ib++) {
            for (ptrdiff_t jb = 0; jb < static_cast<ptrdiff_t>(nt); jb++) {
                const size_t i0 = static_cast<size_t>(ib) * kMTile;
                const size_t j0 = static_cast<size_t>(jb) * kNTile;
                const size_t mi = std::min(kMTile, m - i0);
                const size_t nj = std::min(kNTile, n - j0);
                std::array<float, kMTile * kNTile> acc{};

                for (size_t ii = 0; ii < mi; ii++) {
                    for (size_t jj = 0; jj < nj; jj++) {
                        acc[ii * kNTile + jj] = bias ? bias[j0 + jj] : 0.0f;
                    }
                }

                for (size_t k0 = 0; k0 < k; k0 += kKTile) {
                    const size_t kk = std::min(kKTile, k - k0);
                    for (size_t ii = 0; ii < mi; ii++) {
                        const float *in_row = in + (i0 + ii) * k + k0;
                        for (size_t jj = 0; jj < nj; jj++) {
                            const float *w_row = weight + (j0 + jj) * k + k0;
                            acc[ii * kNTile + jj] += dot_product_f32(in_row, w_row, kk);
                        }
                    }
                }

                for (size_t ii = 0; ii < mi; ii++) {
                    float *out_row = out + (i0 + ii) * n + j0;
                    for (size_t jj = 0; jj < nj; jj++) {
                        out_row[jj] = acc[ii * kNTile + jj];
                    }
                }
            }
        }
        return;
    }
#endif

    for (size_t ib = 0; ib < mt; ib++) {
        for (size_t jb = 0; jb < nt; jb++) {
            const size_t i0 = ib * kMTile;
            const size_t j0 = jb * kNTile;
            const size_t mi = std::min(kMTile, m - i0);
            const size_t nj = std::min(kNTile, n - j0);
            std::array<float, kMTile * kNTile> acc{};

            for (size_t ii = 0; ii < mi; ii++) {
                for (size_t jj = 0; jj < nj; jj++) {
                    acc[ii * kNTile + jj] = bias ? bias[j0 + jj] : 0.0f;
                }
            }

            for (size_t k0 = 0; k0 < k; k0 += kKTile) {
                const size_t kk = std::min(kKTile, k - k0);
                for (size_t ii = 0; ii < mi; ii++) {
                    const float *in_row = in + (i0 + ii) * k + k0;
                    for (size_t jj = 0; jj < nj; jj++) {
                        const float *w_row = weight + (j0 + jj) * k + k0;
                        acc[ii * kNTile + jj] += dot_product_f32(in_row, w_row, kk);
                    }
                }
            }

            for (size_t ii = 0; ii < mi; ii++) {
                float *out_row = out + (i0 + ii) * n + j0;
                for (size_t jj = 0; jj < nj; jj++) {
                    out_row[jj] = acc[ii * kNTile + jj];
                }
            }
        }
    }
}

template <typename T>
void linear_packed_gemv(T *out, const T *in_row, const float *weight, const T *bias, size_t n, size_t k) {
    std::vector<float> input(k);
    convert_to_float(input.data(), in_row, k);

#ifdef _OPENMP
    if (n * k >= kOmpGemvThreshold) {
#pragma omp parallel for schedule(static)
        for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(n); j++) {
            float acc = bias ? llaisys::utils::cast<float>(bias[j]) : 0.0f;
            acc += dot_product_f32(input.data(), weight + static_cast<size_t>(j) * k, k);
            out[j] = llaisys::utils::cast<T>(acc);
        }
        return;
    }
#endif

    for (size_t j = 0; j < n; j++) {
        float acc = bias ? llaisys::utils::cast<float>(bias[j]) : 0.0f;
        acc += dot_product_f32(input.data(), weight + j * k, k);
        out[j] = llaisys::utils::cast<T>(acc);
    }
}

template <typename T>
void linear_packed_gemm(T *out, const T *in, const float *weight, const T *bias, size_t m, size_t n, size_t k) {
    const size_t mt = ceil_div(m, kMTile);
    const size_t nt = ceil_div(n, kNTile);

#ifdef _OPENMP
    if (m * n * k >= kOmpGemmThreshold) {
#pragma omp parallel for collapse(2) schedule(static)
        for (ptrdiff_t ib = 0; ib < static_cast<ptrdiff_t>(mt); ib++) {
            for (ptrdiff_t jb = 0; jb < static_cast<ptrdiff_t>(nt); jb++) {
                const size_t i0 = static_cast<size_t>(ib) * kMTile;
                const size_t j0 = static_cast<size_t>(jb) * kNTile;
                const size_t mi = std::min(kMTile, m - i0);
                const size_t nj = std::min(kNTile, n - j0);
                std::array<float, kMTile * kNTile> acc{};

                for (size_t ii = 0; ii < mi; ii++) {
                    for (size_t jj = 0; jj < nj; jj++) {
                        acc[ii * kNTile + jj] = bias ? llaisys::utils::cast<float>(bias[j0 + jj]) : 0.0f;
                    }
                }

                for (size_t k0 = 0; k0 < k; k0 += kKTile) {
                    const size_t kk = std::min(kKTile, k - k0);
                    std::array<float, kMTile * kKTile> input_tile{};
                    for (size_t ii = 0; ii < mi; ii++) {
                        const T *in_row = in + (i0 + ii) * k + k0;
                        convert_to_float(input_tile.data() + ii * kKTile, in_row, kk);
                    }

                    for (size_t ii = 0; ii < mi; ii++) {
                        const float *in_row = input_tile.data() + ii * kKTile;
                        for (size_t jj = 0; jj < nj; jj++) {
                            const float *w_row = weight + (j0 + jj) * k + k0;
                            acc[ii * kNTile + jj] += dot_product_f32(in_row, w_row, kk);
                        }
                    }
                }

                for (size_t ii = 0; ii < mi; ii++) {
                    T *out_row = out + (i0 + ii) * n + j0;
                    for (size_t jj = 0; jj < nj; jj++) {
                        out_row[jj] = llaisys::utils::cast<T>(acc[ii * kNTile + jj]);
                    }
                }
            }
        }
        return;
    }
#endif

    for (size_t ib = 0; ib < mt; ib++) {
        for (size_t jb = 0; jb < nt; jb++) {
            const size_t i0 = ib * kMTile;
            const size_t j0 = jb * kNTile;
            const size_t mi = std::min(kMTile, m - i0);
            const size_t nj = std::min(kNTile, n - j0);
            std::array<float, kMTile * kNTile> acc{};

            for (size_t ii = 0; ii < mi; ii++) {
                for (size_t jj = 0; jj < nj; jj++) {
                    acc[ii * kNTile + jj] = bias ? llaisys::utils::cast<float>(bias[j0 + jj]) : 0.0f;
                }
            }

            for (size_t k0 = 0; k0 < k; k0 += kKTile) {
                const size_t kk = std::min(kKTile, k - k0);
                std::array<float, kMTile * kKTile> input_tile{};
                for (size_t ii = 0; ii < mi; ii++) {
                    const T *in_row = in + (i0 + ii) * k + k0;
                    convert_to_float(input_tile.data() + ii * kKTile, in_row, kk);
                }

                for (size_t ii = 0; ii < mi; ii++) {
                    const float *in_row = input_tile.data() + ii * kKTile;
                    for (size_t jj = 0; jj < nj; jj++) {
                        const float *w_row = weight + (j0 + jj) * k + k0;
                        acc[ii * kNTile + jj] += dot_product_f32(in_row, w_row, kk);
                    }
                }
            }

            for (size_t ii = 0; ii < mi; ii++) {
                T *out_row = out + (i0 + ii) * n + j0;
                for (size_t jj = 0; jj < nj; jj++) {
                    out_row[jj] = llaisys::utils::cast<T>(acc[ii * kNTile + jj]);
                }
            }
        }
    }
}

#ifdef LLAISYS_USE_OPENBLAS
void apply_bias_f32(float *out, const float *bias, size_t m, size_t n) {
    if (bias == nullptr) {
        return;
    }
    for (size_t i = 0; i < m; i++) {
        float *out_row = out + i * n;
        for (size_t j = 0; j < n; j++) {
            out_row[j] += bias[j];
        }
    }
}
#endif

bool try_openblas_f32(float *out, const float *in, const float *weight, const float *bias, size_t m, size_t n, size_t k) {
#ifdef LLAISYS_USE_OPENBLAS
    if (m == 1 && n >= 2048 && k >= 2048) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(n), static_cast<int>(k), 1.0f, weight, static_cast<int>(k),
                    in, 1, 0.0f, out, 1);
        apply_bias_f32(out, bias, 1, n);
        return true;
    }

    if (m > 1 && m * n * k >= kOmpGemmThreshold) {
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    static_cast<int>(m),
                    static_cast<int>(n),
                    static_cast<int>(k),
                    1.0f,
                    in,
                    static_cast<int>(k),
                    weight,
                    static_cast<int>(k),
                    0.0f,
                    out,
                    static_cast<int>(n));
        apply_bias_f32(out, bias, m, n);
        return true;
    }
#endif
    (void)out;
    (void)in;
    (void)weight;
    (void)bias;
    (void)m;
    (void)n;
    (void)k;
    return false;
}
} // namespace

namespace llaisys::ops::cpu {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    const size_t m = out->shape()[0];
    const size_t n = out->shape()[1];
    const size_t k = in->shape()[1];
    const bool contiguous = out->isContiguous() && in->isContiguous() && weight->isContiguous() && (!bias || bias->isContiguous());

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32: {
        auto *out_ptr = reinterpret_cast<float *>(out->data());
        const auto *in_ptr = reinterpret_cast<const float *>(in->data());
        const auto *weight_ptr = reinterpret_cast<const float *>(weight->data());
        const auto *bias_ptr = bias ? reinterpret_cast<const float *>(bias->data()) : nullptr;

        if (use_legacy_linear() || !contiguous) {
            return linear_reference(out_ptr, in_ptr, weight_ptr, bias_ptr, m, n, k);
        }
        if (try_openblas_f32(out_ptr, in_ptr, weight_ptr, bias_ptr, m, n, k)) {
            return;
        }
        if (m == 1) {
            return linear_f32_gemv(out_ptr, in_ptr, weight_ptr, bias_ptr, n, k);
        }
        return linear_f32_gemm(out_ptr, in_ptr, weight_ptr, bias_ptr, m, n, k);
    }
    case LLAISYS_DTYPE_F16: {
        auto *out_ptr = reinterpret_cast<fp16_t *>(out->data());
        const auto *in_ptr = reinterpret_cast<const fp16_t *>(in->data());
        const auto *weight_ptr = reinterpret_cast<const fp16_t *>(weight->data());
        const auto *bias_ptr = bias ? reinterpret_cast<const fp16_t *>(bias->data()) : nullptr;

        if (use_legacy_linear() || !contiguous) {
            return linear_reference(out_ptr, in_ptr, weight_ptr, bias_ptr, m, n, k);
        }
        const float *packed = weight->getOrCreatePackedLinearF32();
        if (m == 1) {
            return linear_packed_gemv(out_ptr, in_ptr, packed, bias_ptr, n, k);
        }
        return linear_packed_gemm(out_ptr, in_ptr, packed, bias_ptr, m, n, k);
    }
    case LLAISYS_DTYPE_BF16: {
        auto *out_ptr = reinterpret_cast<bf16_t *>(out->data());
        const auto *in_ptr = reinterpret_cast<const bf16_t *>(in->data());
        const auto *weight_ptr = reinterpret_cast<const bf16_t *>(weight->data());
        const auto *bias_ptr = bias ? reinterpret_cast<const bf16_t *>(bias->data()) : nullptr;

        if (use_legacy_linear() || !contiguous) {
            return linear_reference(out_ptr, in_ptr, weight_ptr, bias_ptr, m, n, k);
        }
        const float *packed = weight->getOrCreatePackedLinearF32();
        if (m == 1) {
            return linear_packed_gemv(out_ptr, in_ptr, packed, bias_ptr, n, k);
        }
        return linear_packed_gemm(out_ptr, in_ptr, packed, bias_ptr, m, n, k);
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops::cpu
