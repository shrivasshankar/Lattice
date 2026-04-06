#include "vector.h"
#include "distance.h"
#include "search.h"
#include "hnsw.h"

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
              << std::fixed << std::setprecision(2) << std::setw(10) << ms << " ms\n";
}

void benchmark_hnsw_build(uint32_t num_vectors, uint32_t dim, uint32_t M) {
    auto dataset = lattice::generate_random_vectors(num_vectors, dim);

    auto start = std::chrono::high_resolution_clock::now();
    lattice::HNSWIndex index(dataset, {.M = M, .ef_construction = 200, .seed = 42});
    index.build();
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "  " << std::setw(10) << num_vectors << " vectors | M="
              << std::setw(2) << M << " | "
              << std::fixed << std::setprecision(1) << std::setw(10) << ms << " ms | "
              << "layers=" << index.get_max_layer() << "\n";
}

int main() {
    std::cout << "=== Lattice Vector Search Engine — Week 4 ===\n\n";

    const uint32_t dim = 128;

    // ── HNSW build benchmarks ──────────────────────────────────────────────
    std::cout << "--- HNSW Build Benchmarks (dim=" << dim << ") ---\n";
    benchmark_hnsw_build(1000, dim, 16);
    benchmark_hnsw_build(5000, dim, 16);
    benchmark_hnsw_build(10000, dim, 16);
    benchmark_hnsw_build(50000, dim, 16);

    // ── Effect of M on build time ──────────────────────────────────────────
    std::cout << "\n--- Effect of M on build (10K vectors, dim=" << dim << ") ---\n";
    benchmark_hnsw_build(10000, dim, 4);
    benchmark_hnsw_build(10000, dim, 8);
    benchmark_hnsw_build(10000, dim, 16);
    benchmark_hnsw_build(10000, dim, 32);

    // ── Layer distribution at scale ────────────────────────────────────────
    std::cout << "\n--- Layer Distribution (50K vectors, M=16) ---\n";
    auto dataset = lattice::generate_random_vectors(50000, dim, 42);
    lattice::HNSWIndex index(dataset, {.M = 16, .ef_construction = 200, .seed = 42});
    index.build();

    std::vector<uint32_t> layer_counts(index.get_max_layer() + 1, 0);
    for (uint32_t i = 0; i < 50000; ++i) {
        layer_counts[index.get_node_level(i)]++;
    }
    for (uint32_t l = 0; l <= index.get_max_layer(); ++l) {
        std::cout << "  Layer " << l << ": " << std::setw(6) << layer_counts[l] << " nodes\n";
    }

    // ── Brute-force baseline ───────────────────────────────────────────────
    std::cout << "\n--- Brute-force KNN baseline (dim=128, k=10) ---\n";
    benchmark_brute_force(1000, dim, 10);
    benchmark_brute_force(10000, dim, 10);
    benchmark_brute_force(100000, dim, 10);

    std::cout << "\nDone. HNSW search coming in Week 5.\n";
    return 0;
}
