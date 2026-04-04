#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lattice {

// A dataset of dense float vectors stored contiguously in row-major order.
// All vectors share the same dimensionality.
//
// Memory layout:  [v0_d0, v0_d1, ..., v0_dD, v1_d0, v1_d1, ..., v1_dD, ...]
//                  |-------- vector 0 --------|-------- vector 1 --------| ...
struct VectorDataset {
    uint32_t num_vectors = 0;
    uint32_t dimension   = 0;
    std::vector<float> data;  // size = num_vectors * dimension

    const float* get_vector(uint32_t index) const;
    float* get_vector(uint32_t index);
};

// Binary file format:
//   bytes 0-3:   num_vectors  (little-endian uint32)
//   bytes 4-7:   dimension    (little-endian uint32)
//   bytes 8+:    num_vectors * dimension floats (IEEE 754, little-endian)
VectorDataset load_vectors(const std::string& filepath);
void save_vectors(const std::string& filepath, const VectorDataset& dataset);

// Generate a dataset of random vectors in [0, 1) for testing.
VectorDataset generate_random_vectors(uint32_t num_vectors, uint32_t dimension,
                                      uint32_t seed = 42);

// Print the first `count` vectors to stdout (for debugging).
void print_vectors(const VectorDataset& dataset, uint32_t count = 5);

} // namespace lattice
