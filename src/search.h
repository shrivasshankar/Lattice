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

// Compute recall@k: what fraction of the true k nearest neighbors
// did the approximate search actually find?
// Returns a value in [0.0, 1.0] where 1.0 = perfect recall.
float compute_recall(
    const std::vector<SearchResult>& approximate,
    const std::vector<SearchResult>& ground_truth,
    uint32_t k
);

} // namespace lattice
