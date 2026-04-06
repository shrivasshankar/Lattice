#include "vector.h"
#include "distance.h"
#include "simd_distance.h"
#include "search.h"
#include "hnsw.h"
#include "thread_pool.h"

#include <chrono>
#include <future>
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    std::cout << "=== Lattice Vector Search Engine — Week 8 ===\n\n";

    const uint32_t N = 50000;
    const uint32_t dim = 128;
    const uint32_t k = 10;
    const uint32_t ef_search = 100;
    const uint32_t num_queries = 500;

    auto dataset = lattice::generate_random_vectors(N, dim, 42);

    std::cout << "Building HNSW index (" << N << " vectors, dim=" << dim << ")...\n";
    lattice::HNSWIndex index(dataset, {.M = 16, .ef_construction = 200, .seed = 42,
                                        .distance_fn = lattice::l2_distance_simd});
    auto build_start = std::chrono::high_resolution_clock::now();
    index.build();
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(build_end - build_start).count();
    std::cout << "Build time: " << std::fixed << std::setprecision(1) << build_s << "s\n\n";

    // Generate queries
    std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
    for (uint32_t q = 0; q < num_queries; ++q)
        for (uint32_t d = 0; d < dim; ++d)
            queries[q][d] = static_cast<float>(q * dim + d) / (num_queries * dim);

    // ── Sequential search ──────────────────────────────────────────────────
    auto seq_start = std::chrono::high_resolution_clock::now();
    for (uint32_t q = 0; q < num_queries; ++q) {
        auto results = index.search(queries[q].data(), k, ef_search);
    }
    auto seq_end = std::chrono::high_resolution_clock::now();
    double seq_ms = std::chrono::duration<double, std::milli>(seq_end - seq_start).count();

    // ── Parallel search at different thread counts ─────────────────────────
    std::cout << "--- Search Benchmark (" << num_queries << " queries, k=" << k
              << ", ef=" << ef_search << ") ---\n\n";

    uint32_t hw_threads = std::thread::hardware_concurrency();
    std::cout << std::setw(10) << "threads" << " | "
              << std::setw(12) << "total (ms)" << " | "
              << std::setw(12) << "per query" << " | "
              << std::setw(10) << "speedup" << "\n";
    std::cout << std::string(52, '-') << "\n";

    std::cout << std::setw(10) << 1 << " | "
              << std::setw(12) << std::setprecision(1) << seq_ms << " | "
              << std::setw(10) << std::setprecision(3) << (seq_ms / num_queries) << " ms" << " | "
              << std::setw(10) << "1.0x" << "\n";

    for (uint32_t t : {2u, 4u, hw_threads}) {
        lattice::ThreadPool pool(t);

        auto par_start = std::chrono::high_resolution_clock::now();
        std::vector<std::future<std::vector<lattice::SearchResult>>> futures;
        futures.reserve(num_queries);
        for (uint32_t q = 0; q < num_queries; ++q) {
            futures.push_back(pool.submit([&index, &queries, q, k, ef_search] {
                return index.search(queries[q].data(), k, ef_search);
            }));
        }
        for (auto& f : futures) f.get();
        auto par_end = std::chrono::high_resolution_clock::now();
        double par_ms = std::chrono::duration<double, std::milli>(par_end - par_start).count();

        std::cout << std::setw(10) << t << " | "
                  << std::setw(12) << std::setprecision(1) << par_ms << " | "
                  << std::setw(10) << std::setprecision(3) << (par_ms / num_queries) << " ms" << " | "
                  << std::setw(9) << std::setprecision(1) << (seq_ms / par_ms) << "x" << "\n";
    }

    // ── Throughput summary ─────────────────────────────────────────────────
    double seq_qps = num_queries / (seq_ms / 1000.0);
    std::cout << "\nSingle-thread throughput: " << std::setprecision(0) << seq_qps << " queries/sec\n";

    {
        lattice::ThreadPool pool;
        auto par_start = std::chrono::high_resolution_clock::now();
        std::vector<std::future<std::vector<lattice::SearchResult>>> futures;
        futures.reserve(num_queries);
        for (uint32_t q = 0; q < num_queries; ++q) {
            futures.push_back(pool.submit([&index, &queries, q, k, ef_search] {
                return index.search(queries[q].data(), k, ef_search);
            }));
        }
        for (auto& f : futures) f.get();
        auto par_end = std::chrono::high_resolution_clock::now();
        double par_ms = std::chrono::duration<double, std::milli>(par_end - par_start).count();
        double par_qps = num_queries / (par_ms / 1000.0);
        std::cout << "Multi-thread throughput:  " << std::setprecision(0) << par_qps
                  << " queries/sec (" << pool.num_threads() << " threads)\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
