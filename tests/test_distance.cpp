#include "distance.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// ── L2 distance tests ──────────────────────────────────────────────────────

TEST(L2Distance, IdenticalVectorsReturnZero) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    EXPECT_FLOAT_EQ(lattice::l2_distance(a.data(), a.data(), 3), 0.0f);
}

TEST(L2Distance, KnownValues) {
    std::vector<float> a = {0.0f, 0.0f, 0.0f};
    std::vector<float> b = {3.0f, 4.0f, 0.0f};
    // Squared distance: 9 + 16 + 0 = 25
    EXPECT_FLOAT_EQ(lattice::l2_distance(a.data(), b.data(), 3), 25.0f);
}

TEST(L2Distance, SingleDimension) {
    std::vector<float> a = {5.0f};
    std::vector<float> b = {2.0f};
    // (5-2)^2 = 9
    EXPECT_FLOAT_EQ(lattice::l2_distance(a.data(), b.data(), 1), 9.0f);
}

TEST(L2Distance, Symmetric) {
    std::vector<float> a = {1.0f, 3.0f, 5.0f, 7.0f};
    std::vector<float> b = {2.0f, 4.0f, 6.0f, 8.0f};
    EXPECT_FLOAT_EQ(
        lattice::l2_distance(a.data(), b.data(), 4),
        lattice::l2_distance(b.data(), a.data(), 4)
    );
}

// ── Cosine distance tests ──────────────────────────────────────────────────

TEST(CosineDistance, IdenticalVectorsReturnZero) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    EXPECT_NEAR(lattice::cosine_distance(a.data(), a.data(), 3), 0.0f, 1e-6f);
}

TEST(CosineDistance, OrthogonalVectorsReturnOne) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};
    EXPECT_NEAR(lattice::cosine_distance(a.data(), b.data(), 2), 1.0f, 1e-6f);
}

TEST(CosineDistance, OppositeVectorsReturnTwo) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {-1.0f, 0.0f};
    EXPECT_NEAR(lattice::cosine_distance(a.data(), b.data(), 2), 2.0f, 1e-6f);
}

TEST(CosineDistance, ScaleInvariant) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {2.0f, 4.0f, 6.0f};  // same direction, 2x magnitude
    EXPECT_NEAR(lattice::cosine_distance(a.data(), b.data(), 3), 0.0f, 1e-6f);
}

TEST(CosineDistance, ZeroVectorReturnsZero) {
    std::vector<float> a = {0.0f, 0.0f};
    std::vector<float> b = {1.0f, 2.0f};
    EXPECT_FLOAT_EQ(lattice::cosine_distance(a.data(), b.data(), 2), 0.0f);
}
