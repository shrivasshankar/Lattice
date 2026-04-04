#include "vector.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <stdexcept>

namespace lattice {

// ── VectorDataset accessors ────────────────────────────────────────────────

const float* VectorDataset::get_vector(uint32_t index) const {
    if (index >= num_vectors) {
        throw std::out_of_range("vector index " + std::to_string(index) +
                                " out of range [0, " + std::to_string(num_vectors) + ")");
    }
    return data.data() + static_cast<size_t>(index) * dimension;
}

float* VectorDataset::get_vector(uint32_t index) {
    if (index >= num_vectors) {
        throw std::out_of_range("vector index " + std::to_string(index) +
                                " out of range [0, " + std::to_string(num_vectors) + ")");
    }
    return data.data() + static_cast<size_t>(index) * dimension;
}

// ── File I/O ───────────────────────────────────────────────────────────────

VectorDataset load_vectors(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open file: " + filepath);
    }

    VectorDataset ds;
    file.read(reinterpret_cast<char*>(&ds.num_vectors), sizeof(ds.num_vectors));
    file.read(reinterpret_cast<char*>(&ds.dimension), sizeof(ds.dimension));

    if (!file) {
        throw std::runtime_error("failed to read header from: " + filepath);
    }

    size_t total_floats = static_cast<size_t>(ds.num_vectors) * ds.dimension;
    ds.data.resize(total_floats);
    file.read(reinterpret_cast<char*>(ds.data.data()), total_floats * sizeof(float));

    if (!file) {
        throw std::runtime_error("failed to read vector data from: " + filepath);
    }

    return ds;
}

void save_vectors(const std::string& filepath, const VectorDataset& dataset) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open file for writing: " + filepath);
    }

    file.write(reinterpret_cast<const char*>(&dataset.num_vectors), sizeof(dataset.num_vectors));
    file.write(reinterpret_cast<const char*>(&dataset.dimension), sizeof(dataset.dimension));

    size_t total_floats = static_cast<size_t>(dataset.num_vectors) * dataset.dimension;
    file.write(reinterpret_cast<const char*>(dataset.data.data()), total_floats * sizeof(float));

    if (!file) {
        throw std::runtime_error("failed to write to: " + filepath);
    }
}

// ── Random generation ──────────────────────────────────────────────────────

VectorDataset generate_random_vectors(uint32_t num_vectors, uint32_t dimension,
                                      uint32_t seed) {
    VectorDataset ds;
    ds.num_vectors = num_vectors;
    ds.dimension   = dimension;

    size_t total_floats = static_cast<size_t>(num_vectors) * dimension;
    ds.data.resize(total_floats);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < total_floats; ++i) {
        ds.data[i] = dist(rng);
    }

    return ds;
}

// ── Debug printing ─────────────────────────────────────────────────────────

void print_vectors(const VectorDataset& dataset, uint32_t count) {
    uint32_t to_print = std::min(count, dataset.num_vectors);

    std::cout << "VectorDataset: " << dataset.num_vectors << " vectors, "
              << dataset.dimension << " dimensions\n";

    for (uint32_t i = 0; i < to_print; ++i) {
        const float* v = dataset.get_vector(i);
        std::cout << "  [" << i << "] ";
        for (uint32_t d = 0; d < dataset.dimension; ++d) {
            if (d > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << v[d];
            if (d >= 7 && dataset.dimension > 8) {
                std::cout << ", ... (" << dataset.dimension - d - 1 << " more)";
                break;
            }
        }
        std::cout << "\n";
    }

    if (to_print < dataset.num_vectors) {
        std::cout << "  ... (" << dataset.num_vectors - to_print << " more vectors)\n";
    }
}

} // namespace lattice
