#include "distance.h"
#include "simd_distance.h"
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>

// ── Correctness tests ──────────────────────────────────────────────────────
// SIMD results must match scalar results within floating-point tolerance.
// The tolerance is slightly relaxed because SIMD uses fused multiply-add
// which has different rounding than separate multiply + add.

class SIMDDistanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        a128.resize(128);
        b128.resize(128);
        for (int i = 0; i < 128; ++i) {
            a128[i] = dist(rng);
            b128[i] = dist(rng);
        }

        a3.resize(3);
        b3.resize(3);
        for (int i = 0; i < 3; ++i) {
            a3[i] = dist(rng);
            b3[i] = dist(rng);
        }

        // Odd dimension (not multiple of 4) to test tail handling
        a13.resize(13);
        b13.resize(13);
        for (int i = 0; i < 13; ++i) {
            a13[i] = dist(rng);
            b13[i] = dist(rng);
        }
    }

    std::vector<float> a128, b128;
    std::vector<float> a3, b3;
    std::vector<float> a13, b13;
};

TEST_F(SIMDDistanceTest, SIMDIsAvailable) {
    EXPECT_TRUE(lattice::simd_available());
}

TEST_F(SIMDDistanceTest, L2MatchesScalar_Dim128) {
    float scalar = lattice::l2_distance(a128.data(), b128.data(), 128);
    float simd   = lattice::l2_distance_simd(a128.data(), b128.data(), 128);
    EXPECT_NEAR(simd, scalar, scalar * 1e-5f);
}

TEST_F(SIMDDistanceTest, L2MatchesScalar_Dim3) {
    float scalar = lattice::l2_distance(a3.data(), b3.data(), 3);
    float simd   = lattice::l2_distance_simd(a3.data(), b3.data(), 3);
    EXPECT_NEAR(simd, scalar, std::abs(scalar) * 1e-5f + 1e-7f);
}

TEST_F(SIMDDistanceTest, L2MatchesScalar_OddDim) {
    float scalar = lattice::l2_distance(a13.data(), b13.data(), 13);
    float simd   = lattice::l2_distance_simd(a13.data(), b13.data(), 13);
    EXPECT_NEAR(simd, scalar, std::abs(scalar) * 1e-5f + 1e-7f);
}

TEST_F(SIMDDistanceTest, CosineMatchesScalar_Dim128) {
    float scalar = lattice::cosine_distance(a128.data(), b128.data(), 128);
    float simd   = lattice::cosine_distance_simd(a128.data(), b128.data(), 128);
    EXPECT_NEAR(simd, scalar, 1e-5f);
}

TEST_F(SIMDDistanceTest, CosineMatchesScalar_Dim3) {
    float scalar = lattice::cosine_distance(a3.data(), b3.data(), 3);
    float simd   = lattice::cosine_distance_simd(a3.data(), b3.data(), 3);
    EXPECT_NEAR(simd, scalar, 1e-5f);
}

TEST_F(SIMDDistanceTest, CosineMatchesScalar_OddDim) {
    float scalar = lattice::cosine_distance(a13.data(), b13.data(), 13);
    float simd   = lattice::cosine_distance_simd(a13.data(), b13.data(), 13);
    EXPECT_NEAR(simd, scalar, 1e-5f);
}

TEST_F(SIMDDistanceTest, L2IdenticalVectorsIsZero) {
    float d = lattice::l2_distance_simd(a128.data(), a128.data(), 128);
    EXPECT_FLOAT_EQ(d, 0.0f);
}

TEST_F(SIMDDistanceTest, CosineIdenticalVectorsIsZero) {
    float d = lattice::cosine_distance_simd(a128.data(), a128.data(), 128);
    EXPECT_NEAR(d, 0.0f, 1e-6f);
}

TEST_F(SIMDDistanceTest, L2SingleDimension) {
    float a[] = {3.0f};
    float b[] = {7.0f};
    float d = lattice::l2_distance_simd(a, b, 1);
    EXPECT_FLOAT_EQ(d, 16.0f); // (3-7)^2 = 16
}

// ── Benchmark: SIMD vs Scalar ──────────────────────────────────────────────

TEST_F(SIMDDistanceTest, L2_SIMD_FasterThanScalar) {
    const int iters = 100000;

    // Warm up caches
    for (int i = 0; i < 1000; ++i) {
        lattice::l2_distance(a128.data(), b128.data(), 128);
        lattice::l2_distance_simd(a128.data(), b128.data(), 128);
    }

    auto scalar_start = std::chrono::high_resolution_clock::now();
    volatile float scalar_sink = 0;
    for (int i = 0; i < iters; ++i) {
        scalar_sink = lattice::l2_distance(a128.data(), b128.data(), 128);
    }
    auto scalar_end = std::chrono::high_resolution_clock::now();

    auto simd_start = std::chrono::high_resolution_clock::now();
    volatile float simd_sink = 0;
    for (int i = 0; i < iters; ++i) {
        simd_sink = lattice::l2_distance_simd(a128.data(), b128.data(), 128);
    }
    auto simd_end = std::chrono::high_resolution_clock::now();

    double scalar_us = std::chrono::duration<double, std::micro>(scalar_end - scalar_start).count();
    double simd_us   = std::chrono::duration<double, std::micro>(simd_end - simd_start).count();
    double speedup   = scalar_us / simd_us;

    std::cout << "  L2 benchmark (dim=128, " << iters << " iters):\n"
              << "    scalar: " << std::fixed << std::setprecision(0) << scalar_us << " us\n"
              << "    SIMD:   " << simd_us << " us\n"
              << "    speedup: " << std::setprecision(1) << speedup << "x\n";

    (void)scalar_sink;
    (void)simd_sink;

    EXPECT_GT(speedup, 1.5) << "SIMD should be significantly faster than scalar";
}

TEST_F(SIMDDistanceTest, Cosine_SIMD_FasterThanScalar) {
    const int iters = 100000;

    for (int i = 0; i < 1000; ++i) {
        lattice::cosine_distance(a128.data(), b128.data(), 128);
        lattice::cosine_distance_simd(a128.data(), b128.data(), 128);
    }

    auto scalar_start = std::chrono::high_resolution_clock::now();
    volatile float scalar_sink = 0;
    for (int i = 0; i < iters; ++i) {
        scalar_sink = lattice::cosine_distance(a128.data(), b128.data(), 128);
    }
    auto scalar_end = std::chrono::high_resolution_clock::now();

    auto simd_start = std::chrono::high_resolution_clock::now();
    volatile float simd_sink = 0;
    for (int i = 0; i < iters; ++i) {
        simd_sink = lattice::cosine_distance_simd(a128.data(), b128.data(), 128);
    }
    auto simd_end = std::chrono::high_resolution_clock::now();

    double scalar_us = std::chrono::duration<double, std::micro>(scalar_end - scalar_start).count();
    double simd_us   = std::chrono::duration<double, std::micro>(simd_end - simd_start).count();
    double speedup   = scalar_us / simd_us;

    std::cout << "  Cosine benchmark (dim=128, " << iters << " iters):\n"
              << "    scalar: " << std::fixed << std::setprecision(0) << scalar_us << " us\n"
              << "    SIMD:   " << simd_us << " us\n"
              << "    speedup: " << std::setprecision(1) << speedup << "x\n";

    (void)scalar_sink;
    (void)simd_sink;

    EXPECT_GT(speedup, 1.5) << "SIMD should be significantly faster than scalar";
}
