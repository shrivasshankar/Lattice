#include "vector.h"
#include "distance.h"
#include "search.h"
#include "hnsw.h"

#include <chrono>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== Lattice Vector Search Engine — Week 5 ===\n\n";

    const uint32_t dim = 128;
    const uint32_t k = 10;

    // ── Build index ────────────────────────────────────────────────────────
    const uint32_t N = 50000;
    std::cout << "Building HNSW index: " << N << " vectors, dim=" << dim
              << ", M=16, ef_construction=200\n";

    auto dataset = lattice::generate_random_vectors(N, dim, 42);

    auto build_start = std::chrono::high_resolution_clock::now();
    lattice::HNSWIndex index(dataset, {.M = 16, .ef_construction = 200, .seed = 42});
    index.build();
    auto build_end = std::chrono::high_resolution_clock::now();

    double build_s = std::chrono::duration<double>(build_end - build_start).count();
    std::cout << "Build time: " << std::fixed << std::setprecision(1) << build_s << "s\n";
    std::cout << "Max layer: " << index.get_max_layer() << "\n\n";

    // ── HNSW vs Brute-force comparison ─────────────────────────────────────
    const uint32_t num_queries = 100;
    std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
    for (uint32_t q = 0; q < num_queries; ++q) {
        for (uint32_t d = 0; d < dim; ++d) {
            queries[q][d] = static_cast<float>(q * dim + d) / (num_queries * dim);
        }
    }

    // Brute-force ground truth
    auto bf_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<lattice::SearchResult>> ground_truth(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q) {
        ground_truth[q] = lattice::brute_force_knn(dataset, queries[q].data(), k);
    }
    auto bf_end = std::chrono::high_resolution_clock::now();
    double bf_total_ms = std::chrono::duration<double, std::milli>(bf_end - bf_start).count();

    std::cout << "--- HNSW Search: Recall vs Latency (k=" << k << ", " << N << " vectors) ---\n";
    std::cout << std::setw(12) << "ef_search" << " | "
              << std::setw(12) << "recall@10" << " | "
              << std::setw(12) << "avg latency" << " | "
              << std::setw(12) << "speedup" << "\n";
    std::cout << std::string(56, '-') << "\n";

    uint32_t ef_values[] = {10, 20, 50, 100, 200, 500};

    for (uint32_t ef : ef_values) {
        float total_recall = 0.0f;

        auto hnsw_start = std::chrono::high_resolution_clock::now();
        for (uint32_t q = 0; q < num_queries; ++q) {
            auto results = index.search(queries[q].data(), k, ef);
            total_recall += lattice::compute_recall(results, ground_truth[q], k);
        }
        auto hnsw_end = std::chrono::high_resolution_clock::now();
        double hnsw_total_ms = std::chrono::duration<double, std::milli>(hnsw_end - hnsw_start).count();

        float avg_recall = total_recall / num_queries;
        double avg_latency_ms = hnsw_total_ms / num_queries;
        double bf_avg_ms = bf_total_ms / num_queries;
        double speedup = bf_avg_ms / avg_latency_ms;

        std::cout << std::setw(12) << ef << " | "
                  << std::setw(11) << std::fixed << std::setprecision(1) << (avg_recall * 100) << "%" << " | "
                  << std::setw(10) << std::setprecision(3) << avg_latency_ms << " ms" << " | "
                  << std::setw(10) << std::setprecision(1) << speedup << "x" << "\n";
    }

    double bf_avg = bf_total_ms / num_queries;
    std::cout << std::setw(12) << "brute-force" << " | "
              << std::setw(11) << "100.0%" << " | "
              << std::setw(10) << std::fixed << std::setprecision(3) << bf_avg << " ms" << " | "
              << std::setw(10) << "1.0x" << "\n";

    std::cout << "\nDone.\n";
    return 0;
}
