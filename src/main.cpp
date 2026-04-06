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
    std::cout << std::fixed;
    std::cout << "============================================================\n";
    std::cout << "  Lattice Vector Search Engine\n";
    std::cout << "============================================================\n\n";

    const uint32_t N = 50000;
    const uint32_t dim = 128;
    const uint32_t k = 10;
    const uint32_t num_queries = 200;
    const char* index_file = "lattice_index.bin";

    std::cout << "SIMD: " << (lattice::simd_available() ? "YES" : "no") << "\n\n";

    auto dataset = lattice::generate_random_vectors(N, dim, 42);
    lattice::HNSWConfig cfg{.M = 32, .ef_construction = 400, .seed = 42,
                             .distance_fn = lattice::l2_distance_simd};

    // ── Build or load ──────────────────────────────────────────────────────
    lattice::HNSWIndex index(dataset, cfg);

    std::ifstream probe(index_file, std::ios::binary);
    if (probe.good()) {
        probe.close();
        auto t0 = std::chrono::high_resolution_clock::now();
        index.load(index_file);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Loaded index from disk in " << std::setprecision(1) << ms << " ms\n\n";
    } else {
        std::cout << "Building HNSW index (N=" << N << ", dim=" << dim
                  << ", M=" << cfg.M << ", efC=" << cfg.ef_construction << ")...\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        index.build();
        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Built in " << std::setprecision(1) << s << "s\n";

        auto t2 = std::chrono::high_resolution_clock::now();
        index.save(index_file);
        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << "Saved to disk in " << std::setprecision(1)
                  << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms\n\n";
    }

    // ── Queries + ground truth ─────────────────────────────────────────────
    std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
    for (uint32_t q = 0; q < num_queries; ++q)
        for (uint32_t d = 0; d < dim; ++d)
            queries[q][d] = static_cast<float>(q * dim + d) / (num_queries * dim);

    std::vector<std::vector<lattice::SearchResult>> gt(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q)
        gt[q] = lattice::brute_force_knn(dataset, queries[q].data(), k);

    // ── Recall vs Latency ──────────────────────────────────────────────────
    std::cout << "--- Recall vs Latency (k=" << k << ", " << num_queries << " queries) ---\n\n";
    std::cout << std::setw(10) << "ef_search" << " | "
              << std::setw(10) << "recall@10" << " | "
              << std::setw(10) << "latency" << "\n";
    std::cout << std::string(38, '-') << "\n";

    for (uint32_t ef_s : {50u, 100u, 200u, 500u, 1000u}) {
        float total_recall = 0;
        auto s0 = std::chrono::high_resolution_clock::now();
        for (uint32_t q = 0; q < num_queries; ++q) {
            auto results = index.search(queries[q].data(), k, ef_s);
            total_recall += lattice::compute_recall(results, gt[q], k);
        }
        auto s1 = std::chrono::high_resolution_clock::now();
        double avg_ms = std::chrono::duration<double, std::milli>(s1 - s0).count() / num_queries;

        std::cout << std::setw(10) << ef_s << " | "
                  << std::setw(9) << std::setprecision(1) << (total_recall / num_queries * 100) << "%" << " | "
                  << std::setw(8) << std::setprecision(3) << avg_ms << " ms\n";
    }

    // ── Throughput ─────────────────────────────────────────────────────────
    std::cout << "\n--- Throughput (ef=200) ---\n\n";

    auto seq_start = std::chrono::high_resolution_clock::now();
    for (uint32_t q = 0; q < num_queries; ++q)
        index.search(queries[q].data(), k, 200);
    auto seq_end = std::chrono::high_resolution_clock::now();
    double seq_ms = std::chrono::duration<double, std::milli>(seq_end - seq_start).count();

    lattice::ThreadPool pool;
    auto par_start = std::chrono::high_resolution_clock::now();
    std::vector<std::future<std::vector<lattice::SearchResult>>> futures;
    futures.reserve(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q) {
        futures.push_back(pool.submit([&index, &queries, q, k] {
            return index.search(queries[q].data(), k, 200);
        }));
    }
    for (auto& f : futures) f.get();
    auto par_end = std::chrono::high_resolution_clock::now();
    double par_ms = std::chrono::duration<double, std::milli>(par_end - par_start).count();

    std::cout << "  1 thread:  " << std::setprecision(0) << (num_queries / (seq_ms / 1000.0)) << " qps\n";
    std::cout << "  " << pool.num_threads() << " threads: "
              << (num_queries / (par_ms / 1000.0)) << " qps\n";

    std::remove(index_file);
    std::cout << "\nDone.\n";
    return 0;
}
