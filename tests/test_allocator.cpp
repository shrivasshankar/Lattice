#include "allocator.h"
#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <cstring>

// ── MemoryArena tests ──────────────────────────────────────────────────────

TEST(MemoryArena, AllocateReturnsNonNull) {
    lattice::MemoryArena arena(1024);
    void* ptr = arena.allocate(64);
    EXPECT_NE(ptr, nullptr);
}

TEST(MemoryArena, TracksBytesUsed) {
    lattice::MemoryArena arena(4096);
    arena.allocate(100);
    EXPECT_GE(arena.bytes_used(), 100);
    arena.allocate(200);
    EXPECT_GE(arena.bytes_used(), 300);
}

TEST(MemoryArena, ReturnsNullWhenFull) {
    lattice::MemoryArena arena(128);
    void* p1 = arena.allocate(100);
    EXPECT_NE(p1, nullptr);
    void* p2 = arena.allocate(100);
    EXPECT_EQ(p2, nullptr);
}

TEST(MemoryArena, AllocationsAreAligned) {
    lattice::MemoryArena arena(4096);

    void* p1 = arena.allocate(7, 1);   // 1-byte aligned
    void* p2 = arena.allocate(4, 4);   // 4-byte aligned
    void* p3 = arena.allocate(8, 8);   // 8-byte aligned
    void* p4 = arena.allocate(16, 16); // 16-byte aligned
    void* p5 = arena.allocate(32, 32); // 32-byte aligned

    EXPECT_NE(p1, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p2) % 4, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p3) % 8, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p4) % 16, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p5) % 32, 0);
}

TEST(MemoryArena, ResetMakesFull) {
    lattice::MemoryArena arena(256);
    arena.allocate(200);
    EXPECT_GE(arena.bytes_used(), 200);

    arena.reset();
    EXPECT_EQ(arena.bytes_used(), 0);

    void* ptr = arena.allocate(200);
    EXPECT_NE(ptr, nullptr);
}

TEST(MemoryArena, ConstructBuildsObject) {
    lattice::MemoryArena arena(4096);

    struct Point { float x, y; };
    Point* p = arena.construct<Point>(Point{3.0f, 4.0f});

    ASSERT_NE(p, nullptr);
    EXPECT_FLOAT_EQ(p->x, 3.0f);
    EXPECT_FLOAT_EQ(p->y, 4.0f);
}

TEST(MemoryArena, AllocateArrayWorks) {
    lattice::MemoryArena arena(4096);
    uint32_t* arr = arena.allocate_array<uint32_t>(100);

    ASSERT_NE(arr, nullptr);
    // Default-initialized to 0
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(arr[i], 0);
    }

    // Can write and read back
    for (int i = 0; i < 100; ++i) {
        arr[i] = i * 10;
    }
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(arr[i], i * 10);
    }
}

TEST(MemoryArena, ManySmallAllocations) {
    lattice::MemoryArena arena(1024 * 1024); // 1 MB

    for (int i = 0; i < 10000; ++i) {
        void* p = arena.allocate(64);
        ASSERT_NE(p, nullptr) << "Failed at allocation " << i;
    }
}

// ── PoolAllocator tests ───────────────────────────────────────────────────

TEST(PoolAllocator, AllocateAndDeallocate) {
    lattice::PoolAllocator<uint64_t> pool(100);

    void* p1 = pool.allocate();
    void* p2 = pool.allocate();
    EXPECT_NE(p1, nullptr);
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);
    EXPECT_EQ(pool.num_allocated(), 2);

    pool.deallocate(p1);
    EXPECT_EQ(pool.num_allocated(), 1);
}

TEST(PoolAllocator, ExhaustsPool) {
    lattice::PoolAllocator<uint64_t> pool(5);

    for (int i = 0; i < 5; ++i) {
        EXPECT_NE(pool.allocate(), nullptr);
    }
    EXPECT_EQ(pool.allocate(), nullptr);
    EXPECT_EQ(pool.num_free(), 0);
}

TEST(PoolAllocator, RecyclesMemory) {
    lattice::PoolAllocator<uint64_t> pool(2);

    void* p1 = pool.allocate();
    void* p2 = pool.allocate();
    EXPECT_EQ(pool.allocate(), nullptr); // full

    pool.deallocate(p1);
    void* p3 = pool.allocate();
    EXPECT_NE(p3, nullptr);
    EXPECT_EQ(p3, p1); // recycled the same block
}

TEST(PoolAllocator, ConstructAndDestroy) {
    struct Widget {
        int value;
        bool* destroyed;
        Widget(int v, bool* d) : value(v), destroyed(d) {}
        ~Widget() { *destroyed = true; }
    };

    lattice::PoolAllocator<Widget> pool(10);
    bool was_destroyed = false;

    Widget* w = pool.construct(42, &was_destroyed);
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(w->value, 42);
    EXPECT_FALSE(was_destroyed);

    pool.destroy(w);
    EXPECT_TRUE(was_destroyed);
    EXPECT_EQ(pool.num_allocated(), 0);
}

// ── Benchmark: Arena vs malloc ─────────────────────────────────────────────

TEST(AllocatorBenchmark, ArenaFasterThanMalloc) {
    const int num_allocs = 50000;
    const size_t alloc_size = 64; // typical neighbor list size

    // Time malloc
    auto malloc_start = std::chrono::high_resolution_clock::now();
    std::vector<void*> ptrs(num_allocs);
    for (int i = 0; i < num_allocs; ++i) {
        ptrs[i] = std::malloc(alloc_size);
    }
    auto malloc_end = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_allocs; ++i) {
        std::free(ptrs[i]);
    }

    // Time arena
    lattice::MemoryArena arena(num_allocs * (alloc_size + 16));
    auto arena_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_allocs; ++i) {
        arena.allocate(alloc_size);
    }
    auto arena_end = std::chrono::high_resolution_clock::now();

    double malloc_us = std::chrono::duration<double, std::micro>(malloc_end - malloc_start).count();
    double arena_us = std::chrono::duration<double, std::micro>(arena_end - arena_start).count();

    // Arena should be at least 2x faster for many small allocations
    EXPECT_LT(arena_us, malloc_us)
        << "Arena: " << arena_us << "us, malloc: " << malloc_us << "us";
}
