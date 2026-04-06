#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>
#include <vector>

namespace lattice {

// A bump allocator (arena) that pre-allocates a contiguous block of memory
// and hands out aligned chunks from it. No individual deallocation — the
// entire arena is freed at once when destroyed or reset.
//
// Why this is fast:
//   - One large allocation instead of thousands of small ones
//   - allocate() is just a pointer bump + alignment (no malloc overhead)
//   - Memory is contiguous → cache-friendly access patterns
//   - reset() is instant (just resets the offset)
//
// Why this works for HNSW:
//   - All graph nodes are allocated during build()
//   - Individual nodes are never freed — the whole index is freed at once
//   - This matches arena semantics perfectly

class MemoryArena {
public:
    explicit MemoryArena(size_t capacity_bytes);
    ~MemoryArena();

    // Arenas own their memory — no copying or moving
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;

    // Allocate `bytes` of aligned memory from the arena.
    // Returns nullptr if the arena is full.
    void* allocate(size_t bytes, size_t alignment = alignof(std::max_align_t));

    // Construct a T in the arena using placement new.
    // Forwards arguments to T's constructor.
    template <typename T, typename... Args>
    T* construct(Args&&... args) {
        void* mem = allocate(sizeof(T), alignof(T));
        if (!mem) return nullptr;
        return new (mem) T(std::forward<Args>(args)...);
    }

    // Allocate a contiguous array of T in the arena.
    // Objects are default-initialized.
    template <typename T>
    T* allocate_array(size_t count) {
        void* mem = allocate(sizeof(T) * count, alignof(T));
        if (!mem) return nullptr;
        T* arr = static_cast<T*>(mem);
        for (size_t i = 0; i < count; ++i) {
            new (&arr[i]) T();
        }
        return arr;
    }

    // Reset the arena — all previous allocations become invalid.
    // Does not free the underlying memory, just resets the offset.
    void reset();

    size_t bytes_used() const { return offset_; }
    size_t bytes_remaining() const { return capacity_ - offset_; }
    size_t capacity() const { return capacity_; }

private:
    alignas(64) uint8_t* buffer_;  // 64-byte aligned for cache line friendliness
    size_t capacity_;
    size_t offset_;
};

// Fixed-size pool allocator. Pre-allocates N blocks of a fixed size and
// manages them with a free list. O(1) allocate and deallocate.
//
// Use this when you need many objects of the same size and want to
// recycle memory (unlike the arena which never reclaims individual blocks).

template <typename T>
class PoolAllocator {
public:
    explicit PoolAllocator(size_t max_objects)
        : block_size_(sizeof(T) < sizeof(void*) ? sizeof(void*) : sizeof(T))
        , capacity_(max_objects)
        , num_allocated_(0)
    {
        // Allocate one contiguous block for all objects
        size_t alloc_size = block_size_ * max_objects;
        buffer_ = static_cast<uint8_t*>(::operator new(alloc_size, std::align_val_t{alignof(T)}));

        // Build the free list: each free block points to the next
        free_list_ = nullptr;
        for (size_t i = max_objects; i > 0; --i) {
            void* block = buffer_ + (i - 1) * block_size_;
            *static_cast<void**>(block) = free_list_;
            free_list_ = block;
        }
    }

    ~PoolAllocator() {
        ::operator delete(buffer_, std::align_val_t{alignof(T)});
    }

    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    // Allocate one T-sized block from the pool.
    // Returns nullptr if the pool is exhausted.
    void* allocate() {
        if (!free_list_) return nullptr;
        void* block = free_list_;
        free_list_ = *static_cast<void**>(block);
        num_allocated_++;
        return block;
    }

    // Return a block to the pool.
    void deallocate(void* ptr) {
        *static_cast<void**>(ptr) = free_list_;
        free_list_ = ptr;
        num_allocated_--;
    }

    // Construct a T in a pool block.
    template <typename... Args>
    T* construct(Args&&... args) {
        void* mem = allocate();
        if (!mem) return nullptr;
        return new (mem) T(std::forward<Args>(args)...);
    }

    // Destroy and return a T to the pool.
    void destroy(T* obj) {
        obj->~T();
        deallocate(obj);
    }

    size_t num_allocated() const { return num_allocated_; }
    size_t capacity() const { return capacity_; }
    size_t num_free() const { return capacity_ - num_allocated_; }

private:
    uint8_t* buffer_;
    void* free_list_;
    size_t block_size_;
    size_t capacity_;
    size_t num_allocated_;
};

} // namespace lattice
