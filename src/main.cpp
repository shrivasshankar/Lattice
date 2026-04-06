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
              << std::fixed << std::setprecision(2) << std::setw(10) << ms << " ms | "
              << "nearest dist = " << std::setprecision(4) << results[0].distance << "\n";
}

int main() {
    std::cout << "=== Lattice Vector Search Engine — Week 3 ===\n\n";

    // ── HNSW index build ───────────────────────────────────────────────────
    const uint32_t num_vectors = 10000;
    const uint32_t dim = 128;

    std::cout << "--- HNSW Index Build ---\n";
    std::cout << "Building index: " << num_vectors << " vectors, "
              << dim << " dimensions, M=16, ef_construction=200\n";

    auto dataset = lattice::generate_random_vectors(num_vectors, dim, 42);

    auto start = std::chrono::high_resolution_clock::now();
    lattice::HNSWIndex index(dataset, {.M = 16, .ef_construction = 200, .seed = 42});
    index.build();
    auto end = std::chrono::high_resolution_clock::now();

    double build_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Build time:   " << std::fixed << std::setprecision(1) << build_ms << " ms\n";
    std::cout << "Nodes:        " << index.num_nodes() << "\n";
    std::cout << "Max layer:    " << index.get_max_layer() << "\n";
    std::cout << "Entry point:  " << index.get_entry_point() << "\n";

    // Layer distribution
    std::cout << "\nLayer distribution:\n";
    std::vector<uint32_t> layer_counts(index.get_max_layer() + 1, 0);
    for (uint32_t i = 0; i < num_vectors; ++i) {
        layer_counts[index.get_node_level(i)]++;
    }
    for (uint32_t l = 0; l <= index.get_max_layer(); ++l) {
        std::cout << "  Layer " << l << ": " << layer_counts[l] << " nodes\n";
    }

    // ── Brute-force benchmarks (baseline for comparison) ───────────────────
    std::cout << "\n--- Brute-force KNN baseline (dim=128, k=10) ---\n";
    benchmark_brute_force(1000, dim, 10);
    benchmark_brute_force(10000, dim, 10);
    benchmark_brute_force(100000, dim, 10);

    std::cout << "\nDone. HNSW search coming in Week 5.\n";
    return 0;
}
