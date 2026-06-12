#include "hnsw.h"
#include "search.h"
#include "vector.h"
#include "distance.h"
#include <gtest/gtest.h>

// ── Parallel build correctness ──────────────────────────────────────────────
//
// The parallel build produces a *different but equally good* graph each run:
// insert order is nondeterministic, so edges differ, but recall must be at
// parity with the serial reference. These tests assert the invariants that
// must hold regardless of thread interleaving:
//   - every node is inserted exactly once (no lost tickets, no orphans)
//   - recall matches the serial build within a tolerance band
//   - repeated parallel builds stay correct (races are probabilistic;
//     repetition raises the odds of catching one)

namespace {

float mean_recall(lattice::HNSWIndex& index, const lattice::VectorDataset& ds,
                  const lattice::VectorDataset& queries, uint32_t k, uint32_t ef) {
    float total = 0.0f;
    for (uint32_t q = 0; q < queries.num_vectors; ++q) {
        const float* qv = queries.get_vector(q);
        auto truth = lattice::brute_force_knn(ds, qv, k);
        auto approx = index.search(qv, k, ef);
        total += lattice::compute_recall(approx, truth, k);
    }
    return total / static_cast<float>(queries.num_vectors);
}

} // namespace

TEST(HNSWParallel, AllNodesInsertedAndConnected) {
    auto ds = lattice::generate_random_vectors(3000, 32, 42);
    lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 100, .seed = 42,
                                  .num_threads = 4});
    index.build();

    ASSERT_EQ(index.num_nodes(), ds.num_vectors);

    // No orphans: every node must have at least one edge at layer 0.
    // A node that lost the first-insert race without being wired, or whose
    // reverse edges were lost to a race, would show up here.
    for (uint32_t i = 0; i < ds.num_vectors; ++i) {
        EXPECT_FALSE(index.get_neighbors(i, 0).empty())
            << "node " << i << " has no layer-0 neighbors (orphan)";
    }
}

TEST(HNSWParallel, RecallParityWithSerialBuild) {
    auto ds = lattice::generate_random_vectors(4000, 64, 42);
    auto queries = lattice::generate_random_vectors(30, 64, 1234);

    lattice::HNSWIndex serial(ds, {.M = 16, .ef_construction = 200, .seed = 42,
                                   .num_threads = 1});
    serial.build();

    lattice::HNSWIndex parallel(ds, {.M = 16, .ef_construction = 200, .seed = 42,
                                     .num_threads = 4});
    parallel.build();

    float serial_recall = mean_recall(serial, ds, queries, 10, 100);
    float parallel_recall = mean_recall(parallel, ds, queries, 10, 100);

    // Parity band, not equality: parallel insert order differs, so recall
    // wobbles around the serial value in either direction. A real race
    // (corrupted edges) collapses recall, which this band catches.
    EXPECT_GE(parallel_recall, serial_recall - 0.05f)
        << "parallel recall " << parallel_recall
        << " far below serial " << serial_recall;
    EXPECT_GE(serial_recall, 0.80f);
    EXPECT_GE(parallel_recall, 0.80f);
}

TEST(HNSWParallel, RepeatedBuildsStayCorrect) {
    auto ds = lattice::generate_random_vectors(2000, 32, 7);
    auto queries = lattice::generate_random_vectors(10, 32, 99);

    for (int run = 0; run < 3; ++run) {
        lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 100, .seed = 42,
                                      .num_threads = 8});
        index.build();

        ASSERT_EQ(index.num_nodes(), ds.num_vectors) << "run " << run;
        float recall = mean_recall(index, ds, queries, 10, 100);
        EXPECT_GE(recall, 0.80f) << "run " << run << " recall collapsed";
    }
}

TEST(HNSWParallel, ZeroMeansAllCores) {
    auto ds = lattice::generate_random_vectors(1500, 32, 42);
    lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 100, .seed = 42,
                                  .num_threads = 0});
    index.build();

    EXPECT_EQ(index.num_nodes(), ds.num_vectors);
    std::vector<float> query(32, 0.5f);
    EXPECT_EQ(index.search(query.data(), 10, 50).size(), 10);
}
