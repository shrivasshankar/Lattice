#pragma once

#include "vector.h"
#include "distance.h"

#include <cstdint>
#include <vector>

namespace lattice {

struct SearchResult {
    uint32_t index;
    float distance;
};

// Brute-force k-nearest neighbor search.
// Compares the query against every vector in the dataset.
// Returns the k closest vectors, sorted by distance (closest first).
std::vector<SearchResult> brute_force_knn(
    const VectorDataset& dataset,
    const float* query,
    uint32_t k,
    DistanceFn distance_fn = l2_distance
);

} // namespace lattice
