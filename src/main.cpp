#include "vector.h"
#include "distance.h"
#include "search.h"
#include "hnsw.h"
#include "allocator.h"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    std::cout << "=== Lattice Vector Search Engine — Week 6 ===\n\n";

    // ── Allocator benchmark: Arena vs malloc ───────────────────────────────
    const int num_allocs = 100000;
    const size_t alloc_size = 64;

    // malloc
    auto m_start = std::chrono::high_resolution_clock::now();
    std::vector<void*> ptrs(num_allocs);
    for (int i = 0; i < num_allocs; ++i) {
        ptrs[i] = std::malloc(alloc_size);
    }
    auto m_end = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_allocs; ++i) {
        std::free(ptrs[i]);
    }
    double malloc_us = std::chrono::duration<double, std::micro>(m_end - m_start).count();

    // Arena
    lattice::MemoryArena arena(num_allocs * (alloc_size + 16));
    auto a_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_allocs; ++i) {
        arena.allocate(alloc_size);
    }
    auto a_end = std::chrono::high_resolution_clock::now();
    double arena_us = std::chrono::duration<double, std::micro>(a_end - a_start).count();

    std::cout << "--- Allocator Benchmark (" << num_allocs << " x " << alloc_size << " bytes) ---\n";
    std::cout << "  malloc:  " << std::fixed << std::setprecision(0) << malloc_us << " us\n";
    std::cout << "  arena:   " << arena_us << " us\n";
    std::cout << "  speedup: " << std::setprecision(1) << malloc_us / arena_us << "x\n\n";

    // ── HNSW recall/latency (from Week 5) ──────────────────────────────────
    const uint32_t N = 50000;
    const uint32_t dim = 128;
    const uint32_t k = 10;

    std::cout << "--- HNSW Search (k=" << k << ", " << N << " vectors, dim=" << dim << ") ---\n";

    auto dataset = lattice::generate_random_vectors(N, dim, 42);

    auto build_start = std::chrono::high_resolution_clock::now();
    lattice::HNSWIndex index(dataset, {.M = 16, .ef_construction = 200, .seed = 42});
    index.build();
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(build_end - build_start).count();

    std::cout << "Build time: " << std::setprecision(1) << build_s << "s\n\n";

    const uint32_t num_queries = 100;
    std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
    for (uint32_t q = 0; q < num_queries; ++q) {
        for (uint32_t d = 0; d < dim; ++d) {
            queries[q][d] = static_cast<float>(q * dim + d) / (num_queries * dim);
        }
    }

    // Ground truth
    std::vector<std::vector<lattice::SearchResult>> ground_truth(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q) {
        ground_truth[q] = lattice::brute_force_knn(dataset, queries[q].data(), k);
    }

    std::cout << std::setw(12) << "ef_search" << " | "
              << std::setw(12) << "recall@10" << " | "
              << std::setw(12) << "avg latency" << "\n";
    std::cout << std::string(42, '-') << "\n";

    for (uint32_t ef : {10u, 50u, 100u, 200u, 500u}) {
        float total_recall = 0.0f;
        auto start = std::chrono::high_resolution_clock::now();
        for (uint32_t q = 0; q < num_queries; ++q) {
            auto results = index.search(queries[q].data(), k, ef);
            total_recall += lattice::compute_recall(results, ground_truth[q], k);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / num_queries;

        std::cout << std::setw(12) << ef << " | "
                  << std::setw(11) << std::setprecision(1) << (total_recall / num_queries * 100) << "%" << " | "
                  << std::setw(10) << std::setprecision(3) << avg_ms << " ms\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
