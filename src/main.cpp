#include "vector.h"
#include "distance.h"
#include "search.h"

#include <chrono>
#include <iostream>
#include <iomanip>

void benchmark_brute_force(uint32_t num_vectors, uint32_t dimension, uint32_t k) {
    auto dataset = lattice::generate_random_vectors(num_vectors, dimension);
    std::vector<float> query(dimension, 0.5f);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = lattice::brute_force_knn(dataset, query.data(), k);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "  " << std::setw(10) << num_vectors << " vectors | "
              << std::fixed << std::setprecision(2) << std::setw(10) << ms << " ms | "
              << "nearest dist = " << std::setprecision(4) << results[0].distance << "\n";
}

int main() {
    std::cout << "=== Lattice Vector Search Engine — Week 2 ===\n\n";

    // ── Demo: brute-force search on a small dataset ────────────────────────
    std::cout << "--- Demo: small dataset search ---\n";
    auto dataset = lattice::generate_random_vectors(100, 8, 42);
    std::vector<float> query = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    std::cout << "Query: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]\n";
    std::cout << "Top 5 nearest neighbors (L2):\n";

    auto results = lattice::brute_force_knn(dataset, query.data(), 5);
    for (const auto& r : results) {
        const float* v = dataset.get_vector(r.index);
        std::cout << "  vec[" << std::setw(3) << r.index << "]  dist=" 
                  << std::fixed << std::setprecision(4) << r.distance << "  | ";
        for (uint32_t d = 0; d < 8; ++d) {
            if (d > 0) std::cout << ", ";
            std::cout << std::setprecision(3) << v[d];
        }
        std::cout << "\n";
    }

    // ── Demo: cosine distance ──────────────────────────────────────────────
    std::cout << "\nTop 5 nearest neighbors (Cosine):\n";
    auto cosine_results = lattice::brute_force_knn(dataset, query.data(), 5,
                                                    lattice::cosine_distance);
    for (const auto& r : cosine_results) {
        std::cout << "  vec[" << std::setw(3) << r.index << "]  dist="
                  << std::fixed << std::setprecision(6) << r.distance << "\n";
    }

    // ── Benchmark: brute-force at increasing scale ─────────────────────────
    const uint32_t dim = 128;
    const uint32_t k = 10;

    std::cout << "\n--- Benchmark: brute-force KNN (dim=" << dim << ", k=" << k << ") ---\n";
    benchmark_brute_force(1000, dim, k);
    benchmark_brute_force(10000, dim, k);
    benchmark_brute_force(100000, dim, k);
    benchmark_brute_force(500000, dim, k);

    std::cout << "\nDone.\n";
    return 0;
}
