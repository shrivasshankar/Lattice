#include "simd_distance.h"

#include <cmath>

// ── Platform detection ─────────────────────────────────────────────────────
//
// __ARM_NEON is defined by the compiler on ARM targets (Apple Silicon, etc.)
// __SSE__ is defined on x86 targets with SSE support
//
// We pick one at compile time. If neither is available, we fall back to
// scalar code (same as distance.cpp, just compiled through this file).

#if defined(__ARM_NEON)
    #include <arm_neon.h>
    #define LATTICE_SIMD_NEON 1
#elif defined(__SSE__)
    #include <xmmintrin.h>
    #define LATTICE_SIMD_SSE 1
#endif

namespace lattice {

bool simd_available() {
#if defined(LATTICE_SIMD_NEON) || defined(LATTICE_SIMD_SSE)
    return true;
#else
    return false;
#endif
}

// ── L2 Distance ────────────────────────────────────────────────────────────

#if defined(LATTICE_SIMD_NEON)

// ARM NEON: process 4 floats per instruction, unrolled 4x (16 floats/iter)
//
// Key intrinsics:
//   vld1q_f32    — load 4 floats into a 128-bit register
//   vsubq_f32    — subtract 4 float pairs
//   vfmaq_f32    — fused multiply-accumulate: acc += a * b (1 cycle, not 2)
//   vaddvq_f32   — horizontal sum: reduce 4 lanes to 1 scalar

float l2_distance_simd(const float* a, const float* b, uint32_t dim) {
    // Four separate accumulators to maximize instruction-level parallelism.
    // The CPU can pipeline independent multiply-accumulates across them.
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    uint32_t i = 0;

    // Main loop: 16 floats per iteration (4 registers x 4 floats)
    for (; i + 15 < dim; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t d0 = vsubq_f32(va0, vb0);
        sum0 = vfmaq_f32(sum0, d0, d0);

        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t d1 = vsubq_f32(va1, vb1);
        sum1 = vfmaq_f32(sum1, d1, d1);

        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t d2 = vsubq_f32(va2, vb2);
        sum2 = vfmaq_f32(sum2, d2, d2);

        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        float32x4_t d3 = vsubq_f32(va3, vb3);
        sum3 = vfmaq_f32(sum3, d3, d3);
    }

    // Cleanup: 4 floats at a time
    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t d = vsubq_f32(va, vb);
        sum0 = vfmaq_f32(sum0, d, d);
    }

    // Combine accumulators and reduce to scalar
    float32x4_t total = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    float result = vaddvq_f32(total);

    // Scalar tail for remaining elements (dim not multiple of 4)
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        result += d * d;
    }

    return result;
}

float cosine_distance_simd(const float* a, const float* b, uint32_t dim) {
    float32x4_t dot_sum  = vdupq_n_f32(0.0f);
    float32x4_t norma_sum = vdupq_n_f32(0.0f);
    float32x4_t normb_sum = vdupq_n_f32(0.0f);

    uint32_t i = 0;

    // Main loop: 4 floats per iteration
    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);

        dot_sum   = vfmaq_f32(dot_sum, va, vb);
        norma_sum = vfmaq_f32(norma_sum, va, va);
        normb_sum = vfmaq_f32(normb_sum, vb, vb);
    }

    float dot    = vaddvq_f32(dot_sum);
    float norm_a = vaddvq_f32(norma_sum);
    float norm_b = vaddvq_f32(normb_sum);

    // Scalar tail
    for (; i < dim; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom == 0.0f) return 0.0f;
    return 1.0f - (dot / denom);
}

#elif defined(LATTICE_SIMD_SSE)

// x86 SSE: same idea, different intrinsic names
//   _mm_loadu_ps  — load 4 floats (unaligned)
//   _mm_sub_ps    — subtract 4 floats
//   _mm_mul_ps    — multiply 4 floats
//   _mm_add_ps    — add 4 floats

static inline float hsum_ps(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

float l2_distance_simd(const float* a, const float* b, uint32_t dim) {
    __m128 sum0 = _mm_setzero_ps();
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    __m128 sum3 = _mm_setzero_ps();

    uint32_t i = 0;

    for (; i + 15 < dim; i += 16) {
        __m128 d0 = _mm_sub_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i));
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(d0, d0));

        __m128 d1 = _mm_sub_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4));
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(d1, d1));

        __m128 d2 = _mm_sub_ps(_mm_loadu_ps(a + i + 8), _mm_loadu_ps(b + i + 8));
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(d2, d2));

        __m128 d3 = _mm_sub_ps(_mm_loadu_ps(a + i + 12), _mm_loadu_ps(b + i + 12));
        sum3 = _mm_add_ps(sum3, _mm_mul_ps(d3, d3));
    }

    for (; i + 3 < dim; i += 4) {
        __m128 d = _mm_sub_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i));
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(d, d));
    }

    __m128 total = _mm_add_ps(_mm_add_ps(sum0, sum1), _mm_add_ps(sum2, sum3));
    float result = hsum_ps(total);

    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        result += d * d;
    }

    return result;
}

float cosine_distance_simd(const float* a, const float* b, uint32_t dim) {
    __m128 dot_v  = _mm_setzero_ps();
    __m128 na_v   = _mm_setzero_ps();
    __m128 nb_v   = _mm_setzero_ps();

    uint32_t i = 0;
    for (; i + 3 < dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        dot_v = _mm_add_ps(dot_v, _mm_mul_ps(va, vb));
        na_v  = _mm_add_ps(na_v, _mm_mul_ps(va, va));
        nb_v  = _mm_add_ps(nb_v, _mm_mul_ps(vb, vb));
    }

    float dot    = hsum_ps(dot_v);
    float norm_a = hsum_ps(na_v);
    float norm_b = hsum_ps(nb_v);

    for (; i < dim; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom == 0.0f) return 0.0f;
    return 1.0f - (dot / denom);
}

#else

// Scalar fallback (identical to distance.cpp)
float l2_distance_simd(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float cosine_distance_simd(const float* a, const float* b, uint32_t dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom == 0.0f) return 0.0f;
    return 1.0f - (dot / denom);
}

#endif

} // namespace lattice
