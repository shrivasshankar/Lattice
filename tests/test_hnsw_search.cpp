#include "hnsw.h"
#include "search.h"
#include "vector.h"
#include "distance.h"
#include <gtest/gtest.h>
#include <numeric>

// ── Basic search tests ─────────────────────────────────────────────────────

TEST(HNSWSearch, SearchReturnsKResults) {
    auto ds = lattice::generate_random_vectors(1000, 32, 42);
    lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 100, .seed = 42});
    index.build();

    std::vector<float> query(32, 0.5f);
    auto results = index.search(query.data(), 10, 50);

    EXPECT_EQ(results.size(), 10);
}

TEST(HNSWSearch, SearchResultsSortedByDistance) {
    auto ds = lattice::generate_random_vectors(1000, 32, 42);
    lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 100, .seed = 42});
    index.build();

    std::vector<float> query(32, 0.5f);
    auto results = index.search(query.data(), 20, 100);

    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i - 1].distance, results[i].distance);
    }
}

TEST(HNSWSearch, SearchFindsExactMatch) {
    auto ds = lattice::generate_random_vectors(500, 16, 42);
    lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 200, .seed = 42});
    index.build();

    // Search for a vector that's already in the dataset
    const float* vec_100 = ds.get_vector(100);
    auto results = index.search(vec_100, 1, 50);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].index, 100);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
}

TEST(HNSWSearch, SearchEmptyIndex) {
    lattice::VectorDataset ds;
    ds.num_vectors = 0;
    ds.dimension = 8;
    lattice::HNSWIndex index(ds);

    std::vector<float> query(8, 0.5f);
    auto results = index.search(query.data(), 10, 50);
    EXPECT_TRUE(results.empty());
}

TEST(HNSWSearch, EfSearchClampedToK) {
    auto ds = lattice::generate_random_vectors(200, 16, 42);
    lattice::HNSWIndex index(ds, {.M = 8, .ef_construction = 50, .seed = 42});
    index.build();

    std::vector<float> query(16, 0.5f);
    // ef_search < k should be clamped up to k
    auto results = index.search(query.data(), 10, 5);
    EXPECT_EQ(results.size(), 10);
}

// ── Recall tests ───────────────────────────────────────────────────────────

TEST(HNSWSearch, RecallAbove90AtHighEf) {
    auto ds = lattice::generate_random_vectors(5000, 32, 42);
    lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 200, .seed = 42});
    index.build();

    const uint32_t k = 10;
    const uint32_t num_queries = 50;
    float total_recall = 0.0f;

    for (uint32_t q = 0; q < num_queries; ++q) {
        std::vector<float> query(32);
        for (uint32_t d = 0; d < 32; ++d) {
            query[d] = static_cast<float>(q * 32 + d) / (num_queries * 32);
        }

        auto ground_truth = lattice::brute_force_knn(ds, query.data(), k);
        auto hnsw_results = index.search(query.data(), k, 200);

        total_recall += lattice::compute_recall(hnsw_results, ground_truth, k);
    }

    float avg_recall = total_recall / num_queries;
    EXPECT_GT(avg_recall, 0.90f)
        << "Average recall@" << k << " = " << avg_recall << " (expected > 0.90)";
}

TEST(HNSWSearch, HigherEfMeansHigherRecall) {
    auto ds = lattice::generate_random_vectors(5000, 32, 42);
    lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 200, .seed = 42});
    index.build();

    const uint32_t k = 10;
    const uint32_t num_queries = 30;

    auto avg_recall_at_ef = [&](uint32_t ef) {
        float total = 0.0f;
        for (uint32_t q = 0; q < num_queries; ++q) {
            std::vector<float> query(32);
            for (uint32_t d = 0; d < 32; ++d) {
                query[d] = static_cast<float>(q * 32 + d) / (num_queries * 32);
            }
            auto gt = lattice::brute_force_knn(ds, query.data(), k);
            auto approx = index.search(query.data(), k, ef);
            total += lattice::compute_recall(approx, gt, k);
        }
        return total / num_queries;
    };

    float recall_low = avg_recall_at_ef(10);
    float recall_high = avg_recall_at_ef(200);

    EXPECT_GT(recall_high, recall_low)
        << "recall@ef=200 (" << recall_high << ") should be > recall@ef=10 (" << recall_low << ")";
}

TEST(HNSWSearch, HNSWFasterThanBruteForce) {
    auto ds = lattice::generate_random_vectors(10000, 64, 42);
    lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 200, .seed = 42});
    index.build();

    std::vector<float> query(64, 0.5f);
    const uint32_t k = 10;

    // Time brute force
    auto bf_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        lattice::brute_force_knn(ds, query.data(), k);
    }
    auto bf_end = std::chrono::high_resolution_clock::now();
    double bf_ms = std::chrono::duration<double, std::milli>(bf_end - bf_start).count();

    // Time HNSW
    auto hnsw_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        index.search(query.data(), k, 50);
    }
    auto hnsw_end = std::chrono::high_resolution_clock::now();
    double hnsw_ms = std::chrono::duration<double, std::milli>(hnsw_end - hnsw_start).count();

    EXPECT_LT(hnsw_ms, bf_ms)
        << "HNSW (" << hnsw_ms << "ms) should be faster than brute-force (" << bf_ms << "ms)";
}

// ── Recall utility tests ───────────────────────────────────────────────────

TEST(ComputeRecall, PerfectRecall) {
    std::vector<lattice::SearchResult> results = {{0, 1.0f}, {1, 2.0f}, {2, 3.0f}};
    std::vector<lattice::SearchResult> truth = {{0, 1.0f}, {1, 2.0f}, {2, 3.0f}};
    EXPECT_FLOAT_EQ(lattice::compute_recall(results, truth, 3), 1.0f);
}

TEST(ComputeRecall, ZeroRecall) {
    std::vector<lattice::SearchResult> results = {{10, 1.0f}, {11, 2.0f}};
    std::vector<lattice::SearchResult> truth = {{0, 1.0f}, {1, 2.0f}};
    EXPECT_FLOAT_EQ(lattice::compute_recall(results, truth, 2), 0.0f);
}

TEST(ComputeRecall, PartialRecall) {
    std::vector<lattice::SearchResult> results = {{0, 1.0f}, {5, 2.0f}, {2, 3.0f}};
    std::vector<lattice::SearchResult> truth = {{0, 1.0f}, {1, 2.0f}, {2, 3.0f}};
    EXPECT_NEAR(lattice::compute_recall(results, truth, 3), 2.0f / 3.0f, 1e-6f);
}
