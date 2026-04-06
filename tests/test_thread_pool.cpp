#include "thread_pool.h"
#include "hnsw.h"
#include "vector.h"
#include "distance.h"
#include "simd_distance.h"
#include "search.h"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>

// ── ThreadPool basic tests ─────────────────────────────────────────────────

TEST(ThreadPool, CreatesRequestedThreads) {
    lattice::ThreadPool pool(4);
    EXPECT_EQ(pool.num_threads(), 4);
}

TEST(ThreadPool, DefaultUsesHardwareConcurrency) {
    lattice::ThreadPool pool;
    EXPECT_GE(pool.num_threads(), 1);
}

TEST(ThreadPool, SubmitReturnsCorrectResult) {
    lattice::ThreadPool pool(2);
    auto future = pool.submit([] { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST(ThreadPool, SubmitMultipleTasks) {
    lattice::ThreadPool pool(4);
    std::vector<std::future<int>> futures;

    for (int i = 0; i < 100; ++i) {
        futures.push_back(pool.submit([i] { return i * i; }));
    }

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(futures[i].get(), i * i);
    }
}

TEST(ThreadPool, TasksRunInParallel) {
    lattice::ThreadPool pool(4);
    std::atomic<int> concurrent_count{0};
    std::atomic<int> max_concurrent{0};

    std::vector<std::future<void>> futures;
    for (int i = 0; i < 8; ++i) {
        futures.push_back(pool.submit([&] {
            int current = ++concurrent_count;
            // Track the max number of tasks running at the same time
            int prev = max_concurrent.load();
            while (current > prev && !max_concurrent.compare_exchange_weak(prev, current)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            --concurrent_count;
        }));
    }

    for (auto& f : futures) f.get();
    EXPECT_GE(max_concurrent.load(), 2) << "Tasks should run concurrently";
}

TEST(ThreadPool, HandlesExceptions) {
    lattice::ThreadPool pool(2);
    auto future = pool.submit([] {
        throw std::runtime_error("test error");
        return 0;
    });
    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST(ThreadPool, StressTest) {
    lattice::ThreadPool pool(8);
    std::atomic<int> counter{0};

    std::vector<std::future<void>> futures;
    for (int i = 0; i < 10000; ++i) {
        futures.push_back(pool.submit([&counter] {
            counter.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    for (auto& f : futures) f.get();
    EXPECT_EQ(counter.load(), 10000);
}

// ── Parallel HNSW search benchmark ────────────────────────────────────────

TEST(ParallelSearch, FasterThanSequential) {
    const uint32_t N = 20000;
    const uint32_t dim = 128;
    const uint32_t k = 10;
    const uint32_t num_queries = 200;
    const uint32_t ef_search = 100;

    auto dataset = lattice::generate_random_vectors(N, dim, 42);
    lattice::HNSWIndex index(dataset, {.M = 16, .ef_construction = 200, .seed = 42,
                                        .distance_fn = lattice::l2_distance_simd});
    index.build();

    // Generate queries
    std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
    for (uint32_t q = 0; q < num_queries; ++q)
        for (uint32_t d = 0; d < dim; ++d)
            queries[q][d] = static_cast<float>(q * dim + d) / (num_queries * dim);

    // Sequential search
    auto seq_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<lattice::SearchResult>> seq_results(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q) {
        seq_results[q] = index.search(queries[q].data(), k, ef_search);
    }
    auto seq_end = std::chrono::high_resolution_clock::now();
    double seq_ms = std::chrono::duration<double, std::milli>(seq_end - seq_start).count();

    // Parallel search
    lattice::ThreadPool pool;
    auto par_start = std::chrono::high_resolution_clock::now();
    std::vector<std::future<std::vector<lattice::SearchResult>>> futures;
    for (uint32_t q = 0; q < num_queries; ++q) {
        futures.push_back(pool.submit([&index, &queries, q, k, ef_search] {
            return index.search(queries[q].data(), k, ef_search);
        }));
    }
    std::vector<std::vector<lattice::SearchResult>> par_results(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q) {
        par_results[q] = futures[q].get();
    }
    auto par_end = std::chrono::high_resolution_clock::now();
    double par_ms = std::chrono::duration<double, std::milli>(par_end - par_start).count();

    double speedup = seq_ms / par_ms;

    std::cout << "  Parallel search benchmark (" << num_queries << " queries, "
              << pool.num_threads() << " threads):\n"
              << "    sequential: " << std::fixed << std::setprecision(1) << seq_ms << " ms\n"
              << "    parallel:   " << par_ms << " ms\n"
              << "    speedup:    " << std::setprecision(1) << speedup << "x\n";

    // Parallel results must be identical to sequential
    for (uint32_t q = 0; q < num_queries; ++q) {
        ASSERT_EQ(par_results[q].size(), seq_results[q].size());
        for (size_t i = 0; i < par_results[q].size(); ++i) {
            EXPECT_EQ(par_results[q][i].index, seq_results[q][i].index);
        }
    }

    EXPECT_GT(speedup, 1.5) << "Parallel should be significantly faster";
}
