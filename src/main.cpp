#include "vector.h"
#include <iostream>
#include <filesystem>

int main() {
    std::cout << "=== Lattice Vector Search Engine ===\n\n";

    // Step 1: Generate random vectors
    const uint32_t num_vectors = 1000;
    const uint32_t dimension   = 128;

    std::cout << "Generating " << num_vectors << " random vectors ("
              << dimension << "D)...\n";
    auto dataset = lattice::generate_random_vectors(num_vectors, dimension);

    // Step 2: Save to disk
    const std::string filepath = "test_vectors.bin";
    std::cout << "Saving to " << filepath << "...\n";
    lattice::save_vectors(filepath, dataset);

    // Verify file size matches expectations
    auto file_size = std::filesystem::file_size(filepath);
    size_t expected = 8 + static_cast<size_t>(num_vectors) * dimension * sizeof(float);
    std::cout << "File size: " << file_size << " bytes (expected " << expected << ")\n\n";

    // Step 3: Load back from disk
    std::cout << "Loading from " << filepath << "...\n";
    auto loaded = lattice::load_vectors(filepath);

    // Step 4: Print a few vectors to confirm round-trip
    std::cout << "\nOriginal:\n";
    lattice::print_vectors(dataset, 3);

    std::cout << "\nLoaded:\n";
    lattice::print_vectors(loaded, 3);

    // Step 5: Verify data integrity
    bool match = (dataset.num_vectors == loaded.num_vectors) &&
                 (dataset.dimension == loaded.dimension) &&
                 (dataset.data == loaded.data);

    std::cout << "\nRound-trip integrity check: "
              << (match ? "PASS" : "FAIL") << "\n";

    return match ? 0 : 1;
}
