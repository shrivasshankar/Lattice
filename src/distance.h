#pragma once

#include <cstdint>

namespace lattice {

// Function pointer type for distance functions.
// Takes two vectors (as float pointers) and their shared dimension.
// Returns a non-negative distance where 0 = identical.
using DistanceFn = float (*)(const float* a, const float* b, uint32_t dimension);

// Squared L2 (Euclidean) distance: sum of (a_i - b_i)^2.
// We skip the sqrt — it doesn't change nearest-neighbor ordering
// and avoiding it is faster.
float l2_distance(const float* a, const float* b, uint32_t dimension);

// Cosine distance: 1 - cosine_similarity.
// Range: [0, 2]. 0 = same direction, 1 = orthogonal, 2 = opposite.
float cosine_distance(const float* a, const float* b, uint32_t dimension);

} // namespace lattice
