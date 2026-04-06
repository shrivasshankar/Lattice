#include "allocator.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace lattice {

// ── MemoryArena ────────────────────────────────────────────────────────────

MemoryArena::MemoryArena(size_t capacity_bytes)
    : capacity_(capacity_bytes)
    , offset_(0)
{
    // Allocate the raw buffer with 64-byte alignment for cache friendliness.
    // aligned_alloc requires size to be a multiple of alignment.
    size_t aligned_capacity = (capacity_bytes + 63) & ~size_t(63);
    buffer_ = static_cast<uint8_t*>(std::aligned_alloc(64, aligned_capacity));
    if (!buffer_) {
        throw std::bad_alloc();
    }
    capacity_ = aligned_capacity;
}

MemoryArena::~MemoryArena() {
    std::free(buffer_);
}

void* MemoryArena::allocate(size_t bytes, size_t alignment) {
    // Align the current offset up to the required alignment.
    // Example: if offset=5 and alignment=8, we bump to 8.
    size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

    if (aligned_offset + bytes > capacity_) {
        return nullptr;  // arena is full
    }

    void* ptr = buffer_ + aligned_offset;
    offset_ = aligned_offset + bytes;
    return ptr;
}

void MemoryArena::reset() {
    offset_ = 0;
}

} // namespace lattice
