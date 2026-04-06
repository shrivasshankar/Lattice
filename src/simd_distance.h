#pragma once

#include <cstdint>

namespace lattice {

// SIMD-accelerated distance functions.
// Same signature as the scalar versions in distance.h — drop-in replacements
// usable anywhere a DistanceFn is expected.
//
// On ARM (Apple Silicon): uses NEON intrinsics (4 floats per instruction).
// On x86: uses SSE intrinsics (4 floats per instruction).
// Fallback: scalar loop (identical to distance.h versions).

float l2_distance_simd(const float* a, const float* b, uint32_t dimension);
float cosine_distance_simd(const float* a, const float* b, uint32_t dimension);

// Runtime check: does this platform have SIMD support compiled in?
bool simd_available();

} // namespace lattice
