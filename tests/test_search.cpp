#include "search.h"
#include "distance.h"
#include "vector.h"
#include <gtest/gtest.h>
#include <cmath>

// ── Brute-force KNN tests ──────────────────────────────────────────────────

TEST(BruteForceKNN, FindsExactMatch) {
    // Insert a known vector into the dataset, then search for it.
    // The closest result should be the vector itself (distance 0).
    lattice::VectorDataset ds;
    ds.num_vectors = 5;
    ds.dimension = 3;
    ds.data = {
        0.0f, 0.0f, 0.0f,  // vec 0
        1.0f, 0.0f, 0.0f,  // vec 1
        0.0f, 1.0f, 0.0f,  // vec 2
        0.0f, 0.0f, 1.0f,  // vec 3
        1.0f, 1.0f, 1.0f,  // vec 4
    };

    std::vector<float> query = {1.0f, 1.0f, 1.0f};
    auto results = lattice::brute_force_knn(ds, query.data(), 1);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].index, 4);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
}

TEST(BruteForceKNN, ReturnsKResults) {
    auto ds = lattice::generate_random_vectors(100, 16, 42);
    std::vector<float> query(16, 0.5f);

    auto results = lattice::brute_force_knn(ds, query.data(), 10);
    EXPECT_EQ(results.size(), 10);
}

TEST(BruteForceKNN, ResultsSortedByDistance) {
    auto ds = lattice::generate_random_vectors(200, 32, 99);
    std::vector<float> query(32, 0.0f);

    auto results = lattice::brute_force_knn(ds, query.data(), 20);

    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i - 1].distance, results[i].distance);
    }
}

TEST(BruteForceKNN, KLargerThanDataset) {
    auto ds = lattice::generate_random_vectors(5, 4, 42);
    std::vector<float> query = {0.5f, 0.5f, 0.5f, 0.5f};

    auto results = lattice::brute_force_knn(ds, query.data(), 100);
    EXPECT_EQ(results.size(), 5);
}

TEST(BruteForceKNN, KZeroReturnsEmpty) {
    auto ds = lattice::generate_random_vectors(10, 4, 42);
    std::vector<float> query = {0.0f, 0.0f, 0.0f, 0.0f};

    auto results = lattice::brute_force_knn(ds, query.data(), 0);
    EXPECT_TRUE(results.empty());
}

TEST(BruteForceKNN, EmptyDatasetReturnsEmpty) {
    lattice::VectorDataset ds;
    ds.num_vectors = 0;
    ds.dimension = 4;

    std::vector<float> query = {1.0f, 2.0f, 3.0f, 4.0f};
    auto results = lattice::brute_force_knn(ds, query.data(), 5);
    EXPECT_TRUE(results.empty());
}

TEST(BruteForceKNN, WorksWithCosineDistance) {
    lattice::VectorDataset ds;
    ds.num_vectors = 3;
    ds.dimension = 2;
    ds.data = {
        1.0f,  0.0f,   // vec 0: points right
        0.0f,  1.0f,   // vec 1: points up
        0.7f,  0.7f,   // vec 2: points diagonal (closest to query)
    };

    std::vector<float> query = {1.0f, 1.0f};  // diagonal query
    auto results = lattice::brute_force_knn(ds, query.data(), 1, lattice::cosine_distance);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].index, 2);
}

TEST(BruteForceKNN, BruteForceIsExact) {
    // Verify brute force finds the true nearest neighbor by manually
    // computing all distances and checking the minimum matches.
    auto ds = lattice::generate_random_vectors(1000, 64, 777);
    std::vector<float> query(64, 0.5f);

    auto results = lattice::brute_force_knn(ds, query.data(), 1);

    // Manually find the closest vector
    float min_dist = std::numeric_limits<float>::max();
    uint32_t min_idx = 0;
    for (uint32_t i = 0; i < ds.num_vectors; ++i) {
        float d = lattice::l2_distance(ds.get_vector(i), query.data(), ds.dimension);
        if (d < min_dist) {
            min_dist = d;
            min_idx = i;
        }
    }

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].index, min_idx);
    EXPECT_FLOAT_EQ(results[0].distance, min_dist);
}
