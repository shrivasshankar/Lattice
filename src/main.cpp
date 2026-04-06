#include "vector.h"
#include "distance.h"
#include "simd_distance.h"
#include "search.h"
#include "hnsw.h"
#include "thread_pool.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <future>
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    std::cout << "=== Lattice Vector Search Engine ===\n\n";

    const uint32_t N = 50000;
    const uint32_t dim = 128;
    const uint32_t k = 10;
    const uint32_t num_queries = 500;
    const char* index_file = "lattice_index.bin";

    std::cout << std::fixed;
    auto dataset = lattice::generate_random_vectors(N, dim, 42);
    lattice::HNSWConfig cfg{.M = 16, .ef_construction = 200, .seed = 42,
                             .distance_fn = lattice::l2_distance_simd};

    // ── Build or load index ────────────────────────────────────────────────
    lattice::HNSWIndex index(dataset, cfg);

    std::ifstream test_file(index_file, std::ios::binary);
    if (test_file.good()) {
        test_file.close();
        std::cout << "Loading index from " << index_file << "...\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        index.load(index_file);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Loaded in " << std::fixed << std::setprecision(1) << ms << " ms ("
                  << index.num_nodes() << " nodes)\n\n";
    } else {
        std::cout << "Building HNSW index (" << N << " vectors, dim=" << dim << ")...\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        index.build();
        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Built in " << std::setprecision(1) << s << "s\n";

        std::cout << "Saving index to " << index_file << "...\n";
        auto t2 = std::chrono::high_resolution_clock::now();
        index.save(index_file);
        auto t3 = std::chrono::high_resolution_clock::now();
        double save_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::cout << "Saved in " << std::setprecision(1) << save_ms << " ms\n\n";
    }

    // ── Generate queries ───────────────────────────────────────────────────
    std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
    for (uint32_t q = 0; q < num_queries; ++q)
        for (uint32_t d = 0; d < dim; ++d)
            queries[q][d] = static_cast<float>(q * dim + d) / (num_queries * dim);

    // ── Recall vs latency curve ────────────────────────────────────────────
    std::vector<std::vector<lattice::SearchResult>> gt(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q)
        gt[q] = lattice::brute_force_knn(dataset, queries[q].data(), k);

    std::cout << "--- Recall vs Latency (k=" << k << ") ---\n\n";
    std::cout << std::setw(12) << "ef_search" << " | "
              << std::setw(12) << "recall@10" << " | "
              << std::setw(12) << "avg latency" << "\n";
    std::cout << std::string(42, '-') << "\n";

    for (uint32_t ef : {10u, 50u, 100u, 200u, 500u}) {
        float total_recall = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint32_t q = 0; q < num_queries; ++q) {
            auto results = index.search(queries[q].data(), k, ef);
            total_recall += lattice::compute_recall(results, gt[q], k);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / num_queries;

        std::cout << std::setw(12) << ef << " | "
                  << std::setw(11) << std::setprecision(1) << (total_recall / num_queries * 100) << "%" << " | "
                  << std::setw(10) << std::setprecision(3) << avg_ms << " ms\n";
    }

    // ── Throughput: sequential vs parallel ─────────────────────────────────
    std::cout << "\n--- Throughput ---\n\n";

    auto seq_start = std::chrono::high_resolution_clock::now();
    for (uint32_t q = 0; q < num_queries; ++q) {
        index.search(queries[q].data(), k, 100);
    }
    auto seq_end = std::chrono::high_resolution_clock::now();
    double seq_ms = std::chrono::duration<double, std::milli>(seq_end - seq_start).count();

    lattice::ThreadPool pool;
    auto par_start = std::chrono::high_resolution_clock::now();
    std::vector<std::future<std::vector<lattice::SearchResult>>> futures;
    futures.reserve(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q) {
        futures.push_back(pool.submit([&index, &queries, q, k] {
            return index.search(queries[q].data(), k, 100);
        }));
    }
    for (auto& f : futures) f.get();
    auto par_end = std::chrono::high_resolution_clock::now();
    double par_ms = std::chrono::duration<double, std::milli>(par_end - par_start).count();

    std::cout << "  Sequential: " << std::setprecision(0) << (num_queries / (seq_ms / 1000.0))
              << " queries/sec (1 thread)\n";
    std::cout << "  Parallel:   " << (num_queries / (par_ms / 1000.0))
              << " queries/sec (" << pool.num_threads() << " threads)\n";

    // Cleanup
    std::remove(index_file);

    std::cout << "\nDone.\n";
    return 0;
}
