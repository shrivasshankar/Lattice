// Scaling benchmark: build the same dataset at increasing thread counts,
// report build time, speedup vs serial, and recall parity. This is the
// measured scaling curve — the numbers behind the "Nx speedup" claim.
//
// Usage: ./scaling_benchmark [num_vectors] [dim]

#include "hnsw.h"
#include "vector.h"
#include "search.h"
#include "simd_distance.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace lattice;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    uint32_t n   = (argc > 1) ? std::atoi(argv[1]) : 50000;
    uint32_t dim = (argc > 2) ? std::atoi(argv[2]) : 128;

    printf("Scaling benchmark: %u vectors x %u dims\n", n, dim);
    printf("Hardware concurrency: %u\n\n", std::thread::hardware_concurrency());

    auto data    = generate_random_vectors(n, dim, 42);
    auto queries = generate_random_vectors(100, dim, 1234);

    // Ground truth (computed once, reused for every recall check).
    std::vector<std::vector<SearchResult>> truth(queries.num_vectors);
    for (uint32_t q = 0; q < queries.num_vectors; ++q) {
        truth[q] = brute_force_knn(data, queries.get_vector(q), 10, l2_distance_simd);
    }

    std::vector<uint32_t> thread_counts = {1, 2, 4, 8};
    uint32_t hw = std::thread::hardware_concurrency();
    if (hw > 8) thread_counts.push_back(hw);

    double serial_time = 0.0;

    printf("%-8s %-12s %-9s %-12s\n", "threads", "build(s)", "speedup", "recall@10");
    printf("---------------------------------------------\n");

    for (uint32_t t : thread_counts) {
        HNSWConfig cfg;
        cfg.distance_fn = l2_distance_simd;
        cfg.num_threads = t;

        HNSWIndex index(data, cfg);
        auto t0 = Clock::now();
        index.build();
        double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        if (t == 1) serial_time = secs;

        float recall = 0.0f;
        for (uint32_t q = 0; q < queries.num_vectors; ++q) {
            auto approx = index.search(queries.get_vector(q), 10, 200);
            recall += compute_recall(approx, truth[q], 10);
        }
        recall /= static_cast<float>(queries.num_vectors);

        printf("%-8u %-12.2f %-9.2f %-12.3f\n",
               t, secs, serial_time / secs, recall);
    }

    return 0;
}
