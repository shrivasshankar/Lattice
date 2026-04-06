#include "vector.h"
#include "distance.h"
#include "simd_distance.h"
#include "search.h"
#include "hnsw.h"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

int main() {
    std::cout << "=== Lattice Vector Search Engine — Week 7 ===\n\n";

    // ── 1. SIMD vs Scalar distance benchmark ──────────────────────────────
    std::cout << "--- Distance Benchmark (dim=128, 1M computations) ---\n";
    std::cout << "  SIMD available: " << (lattice::simd_available() ? "YES" : "no") << "\n\n";

    const uint32_t dim = 128;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> va(dim), vb(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        va[i] = dist(rng);
        vb[i] = dist(rng);
    }

    const int bench_iters = 1000000;

    // L2 scalar
    auto t0 = std::chrono::high_resolution_clock::now();
    volatile float sink = 0;
    for (int i = 0; i < bench_iters; ++i) {
        sink = lattice::l2_distance(va.data(), vb.data(), dim);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double l2_scalar_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    (void)sink;

    // L2 SIMD
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        sink = lattice::l2_distance_simd(va.data(), vb.data(), dim);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double l2_simd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    (void)sink;

    // Cosine scalar
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        sink = lattice::cosine_distance(va.data(), vb.data(), dim);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double cos_scalar_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    (void)sink;

    // Cosine SIMD
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        sink = lattice::cosine_distance_simd(va.data(), vb.data(), dim);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double cos_simd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    (void)sink;

    std::cout << std::fixed;
    std::cout << "  L2:     scalar " << std::setprecision(1) << l2_scalar_ms << " ms"
              << "  |  SIMD " << l2_simd_ms << " ms"
              << "  |  " << std::setprecision(1) << l2_scalar_ms / l2_simd_ms << "x speedup\n";
    std::cout << "  Cosine: scalar " << cos_scalar_ms << " ms"
              << "  |  SIMD " << cos_simd_ms << " ms"
              << "  |  " << cos_scalar_ms / cos_simd_ms << "x speedup\n\n";

    // ── 2. HNSW with scalar vs SIMD ───────────────────────────────────────
    const uint32_t N = 50000;
    const uint32_t k = 10;

    std::cout << "--- HNSW End-to-End: Scalar vs SIMD (N=" << N << ", dim=" << dim << ") ---\n\n";

    auto dataset = lattice::generate_random_vectors(N, dim, 42);

    // Build + search with SCALAR
    {
        lattice::HNSWConfig cfg{.M = 16, .ef_construction = 200, .seed = 42, .distance_fn = lattice::l2_distance};
        auto build_start = std::chrono::high_resolution_clock::now();
        lattice::HNSWIndex index(dataset, cfg);
        index.build();
        auto build_end = std::chrono::high_resolution_clock::now();
        double build_s = std::chrono::duration<double>(build_end - build_start).count();

        // Queries
        const uint32_t nq = 100;
        std::vector<std::vector<float>> queries(nq, std::vector<float>(dim));
        for (uint32_t q = 0; q < nq; ++q)
            for (uint32_t d = 0; d < dim; ++d)
                queries[q][d] = static_cast<float>(q * dim + d) / (nq * dim);

        std::vector<std::vector<lattice::SearchResult>> gt(nq);
        for (uint32_t q = 0; q < nq; ++q)
            gt[q] = lattice::brute_force_knn(dataset, queries[q].data(), k);

        auto search_start = std::chrono::high_resolution_clock::now();
        float total_recall = 0;
        for (uint32_t q = 0; q < nq; ++q) {
            auto results = index.search(queries[q].data(), k, 100);
            total_recall += lattice::compute_recall(results, gt[q], k);
        }
        auto search_end = std::chrono::high_resolution_clock::now();
        double search_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();

        std::cout << "  SCALAR:  build " << std::setprecision(1) << build_s << "s"
                  << "  |  search " << std::setprecision(1) << search_ms << " ms (" << nq << " queries)"
                  << "  |  recall " << std::setprecision(1) << (total_recall / nq * 100) << "%\n";
    }

    // Build + search with SIMD
    {
        lattice::HNSWConfig cfg{.M = 16, .ef_construction = 200, .seed = 42, .distance_fn = lattice::l2_distance_simd};
        auto build_start = std::chrono::high_resolution_clock::now();
        lattice::HNSWIndex index(dataset, cfg);
        index.build();
        auto build_end = std::chrono::high_resolution_clock::now();
        double build_s = std::chrono::duration<double>(build_end - build_start).count();

        const uint32_t nq = 100;
        std::vector<std::vector<float>> queries(nq, std::vector<float>(dim));
        for (uint32_t q = 0; q < nq; ++q)
            for (uint32_t d = 0; d < dim; ++d)
                queries[q][d] = static_cast<float>(q * dim + d) / (nq * dim);

        std::vector<std::vector<lattice::SearchResult>> gt(nq);
        for (uint32_t q = 0; q < nq; ++q)
            gt[q] = lattice::brute_force_knn(dataset, queries[q].data(), k);

        auto search_start = std::chrono::high_resolution_clock::now();
        float total_recall = 0;
        for (uint32_t q = 0; q < nq; ++q) {
            auto results = index.search(queries[q].data(), k, 100);
            total_recall += lattice::compute_recall(results, gt[q], k);
        }
        auto search_end = std::chrono::high_resolution_clock::now();
        double search_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();

        std::cout << "  SIMD:    build " << std::setprecision(1) << build_s << "s"
                  << "  |  search " << std::setprecision(1) << search_ms << " ms (" << nq << " queries)"
                  << "  |  recall " << std::setprecision(1) << (total_recall / nq * 100) << "%\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
