#include "vector.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Exports the exact dataset and queries that Lattice benchmarks against,
// so an external tool (e.g. the FAISS comparison) can load byte-identical
// vectors instead of regenerating its own. This is what makes the
// head-to-head "same dataset, same queries" comparison actually true.
//
// Usage: export_dataset <N> [num_queries] [dim]
// Writes:  dataset_<N>.bin   (N x dim float vectors, seed=42, uniform [0,1))
//          queries_<num_queries>.bin  (deterministic query vectors)
//
// File format matches lattice::save_vectors:
//   uint32 count, uint32 dim, then count*dim little-endian float32.

int main(int argc, char** argv) {
    uint32_t N = (argc > 1) ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 50000;
    uint32_t num_queries = (argc > 2) ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 200;
    uint32_t dim = (argc > 3) ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 128;

    // Dataset: identical to main.cpp / tune_recall.cpp (seed=42, uniform [0,1)).
    lattice::VectorDataset dataset = lattice::generate_random_vectors(N, dim, 42);
    std::string dataset_file = "dataset_" + std::to_string(N) + ".bin";
    lattice::save_vectors(dataset_file, dataset);

    // Queries: identical formula to main.cpp / tune_recall.cpp / faiss_comparison.py.
    lattice::VectorDataset queries;
    queries.num_vectors = num_queries;
    queries.dimension = dim;
    queries.data.resize(static_cast<size_t>(num_queries) * dim);
    for (uint32_t q = 0; q < num_queries; ++q)
        for (uint32_t d = 0; d < dim; ++d)
            queries.data[static_cast<size_t>(q) * dim + d] =
                static_cast<float>(q * dim + d) / (num_queries * dim);

    std::string queries_file = "queries_" + std::to_string(num_queries) + ".bin";
    lattice::save_vectors(queries_file, queries);

    std::cout << "Wrote " << dataset_file << " (" << N << " x " << dim << ")\n";
    std::cout << "Wrote " << queries_file << " (" << num_queries << " x " << dim << ")\n";
    return 0;
}
