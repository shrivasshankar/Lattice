#include "vector.h"
#include "distance.h"
#include "simd_distance.h"
#include "search.h"
#include "hnsw.h"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    std::cout << std::fixed;

    const uint32_t N = 50000;
    const uint32_t dim = 128;
    const uint32_t k = 10;
    const uint32_t num_queries = 200;

    auto dataset = lattice::generate_random_vectors(N, dim, 42);

    // Generate queries (use vectors NOT in the dataset for realistic test)
    std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
    for (uint32_t q = 0; q < num_queries; ++q)
        for (uint32_t d = 0; d < dim; ++d)
            queries[q][d] = static_cast<float>(q * dim + d) / (num_queries * dim);

    // Ground truth with scalar (guaranteed correct)
    std::vector<std::vector<lattice::SearchResult>> gt(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q)
        gt[q] = lattice::brute_force_knn(dataset, queries[q].data(), k);

    std::cout << "=== Parameter Sweep (N=" << N << ", dim=" << dim << ", k=" << k << ") ===\n\n";
    std::cout << std::setw(4) << "M" << " | "
              << std::setw(6) << "efC" << " | "
              << std::setw(8) << "efS" << " | "
              << std::setw(10) << "recall@10" << " | "
              << std::setw(10) << "build(s)" << " | "
              << std::setw(10) << "query(ms)" << "\n";
    std::cout << std::string(62, '-') << "\n";

    for (uint32_t M : {16u, 24u, 32u, 48u}) {
        for (uint32_t ef_c : {200u, 400u}) {
            lattice::HNSWConfig cfg{.M = M, .ef_construction = ef_c, .seed = 42,
                                     .distance_fn = lattice::l2_distance_simd};

            auto t0 = std::chrono::high_resolution_clock::now();
            lattice::HNSWIndex index(dataset, cfg);
            index.build();
            auto t1 = std::chrono::high_resolution_clock::now();
            double build_s = std::chrono::duration<double>(t1 - t0).count();

            for (uint32_t ef_s : {50u, 100u, 200u, 500u}) {
                float total_recall = 0;
                auto s0 = std::chrono::high_resolution_clock::now();
                for (uint32_t q = 0; q < num_queries; ++q) {
                    auto results = index.search(queries[q].data(), k, ef_s);
                    total_recall += lattice::compute_recall(results, gt[q], k);
                }
                auto s1 = std::chrono::high_resolution_clock::now();
                double query_ms = std::chrono::duration<double, std::milli>(s1 - s0).count() / num_queries;
                float avg_recall = total_recall / num_queries * 100;

                std::cout << std::setw(4) << M << " | "
                          << std::setw(6) << ef_c << " | "
                          << std::setw(8) << ef_s << " | "
                          << std::setw(9) << std::setprecision(1) << avg_recall << "%" << " | "
                          << std::setw(10) << std::setprecision(1) << build_s << " | "
                          << std::setw(9) << std::setprecision(3) << query_ms << "\n";
            }
        }
    }

    return 0;
}
