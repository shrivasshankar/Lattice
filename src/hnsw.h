#pragma once

#include "vector.h"
#include "distance.h"
#include "search.h"
#include "allocator.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace lattice {

struct HNSWConfig {
    uint32_t M = 32;                // max connections per node per layer
    uint32_t ef_construction = 400; // candidate list size during insert
    uint32_t seed = 42;
    DistanceFn distance_fn = l2_distance;
    uint32_t num_threads = 1;       // build parallelism: 1 = serial (the
                                    // reference path), 0 = all hardware cores

    // Neighbor selection during build:
    //   false = closest-M (fastest build; the default)
    //   true  = diversity heuristic (HNSW paper Algorithm 4): higher recall
    //           at high ef, but ~2-2.5x slower build from the extra pairwise
    //           distance evaluations. A recall-vs-build-time trade.
    bool use_diversity_heuristic = false;
};

class HNSWIndex {
public:
    explicit HNSWIndex(const VectorDataset& dataset, HNSWConfig config = {});

    // Insert a single vector into the index by its ID in the dataset.
    // Draws the node's level from the index's RNG.
    void insert(uint32_t vector_id);

    // Insert with a pre-assigned level. Levels are drawn up front in
    // build() so that insertion itself never touches shared RNG state —
    // a prerequisite for running inserts on multiple threads.
    void insert(uint32_t vector_id, uint32_t node_level);

    // Insert all vectors in the dataset.
    void build();

    // Search the index for the k nearest neighbors of a query vector.
    // ef_search controls quality vs speed (higher = better recall, slower).
    // Must be >= k.
    std::vector<SearchResult> search(
        const float* query,
        uint32_t k,
        uint32_t ef_search = 50
    );

    // Save the index graph structure to a binary file.
    // Only saves the graph (node levels, neighbor lists) — not the vectors.
    // The same dataset must be provided when loading.
    void save(const std::string& filename) const;

    // Load a previously saved index. Restores all graph structure.
    // The dataset passed to the constructor must match the one used at build time.
    void load(const std::string& filename);

    // Getters for testing and inspection
    uint32_t get_max_layer() const {
        std::lock_guard<std::mutex> lock(entry_mutex_);
        return max_layer_;
    }
    uint32_t get_entry_point() const {
        std::lock_guard<std::mutex> lock(entry_mutex_);
        return entry_point_;
    }
    uint32_t num_nodes() const { return num_inserted_.load(); }
    // Returns a copy of a node's neighbor IDs at the given layer. (Copy, not
    // a reference: the underlying storage is a fixed-capacity arena span.)
    std::vector<uint32_t> get_neighbors(uint32_t node_id, uint32_t layer) const;
    uint32_t get_node_level(uint32_t node_id) const;

private:
    // A node's neighbor list at one layer. `data` is backed by the graph
    // arena and fixed-capacity — it never reallocates, so a concurrent search
    // can read it (under the stripe lock) while an insert mutates `count`,
    // with no use-after-free from a vector growing under the reader. `count`
    // is the live length; slots [count, capacity) are unused.
    struct NeighborList {
        uint32_t* data = nullptr;
        uint32_t count = 0;
        uint32_t capacity = 0;
    };

    struct Node {
        std::vector<NeighborList> neighbors;  // one entry per layer 0..level
        uint32_t level = 0;   // highest layer this node appears in
        bool inserted = false;
    };

    // Assign a random layer to a new node.
    // P(level >= l) decreases exponentially, like a skip list.
    uint32_t random_level();

    // Greedy search within one layer of the graph.
    // Starting from entry_ids, finds the ef closest nodes to query
    // by expanding through neighbor links.
    std::vector<SearchResult> search_layer(
        const float* query,
        const std::vector<uint32_t>& entry_ids,
        uint32_t ef,
        uint32_t layer
    );

    // Pick the best neighbors from a candidate set.
    // Simple strategy: take the closest max_neighbors candidates.
    std::vector<uint32_t> select_neighbors(
        const std::vector<SearchResult>& candidates,
        uint32_t max_neighbors
    );

    const VectorDataset& dataset_;
    HNSWConfig config_;

    std::vector<Node> nodes_;

    // entry_point_ and max_layer_ form an invariant pair: search descends
    // from max_layer_ starting at entry_point_, so they must never be
    // observed mid-update. Both are guarded by entry_mutex_; readers take
    // a snapshot of both under the lock and work from locals.
    // (mutable so const getters can lock.)
    mutable std::mutex entry_mutex_;
    uint32_t entry_point_ = 0;
    uint32_t max_layer_ = 0;
    // True once any node has become the entry point. Guarded by
    // entry_mutex_; the first-insert decision is check-and-act under
    // the lock so two concurrent first inserts cannot both win
    // (the loser would return unwired — a permanently orphaned node).
    bool has_entry_ = false;

    // Striped locks guarding per-node state (neighbor lists, level,
    // inserted flag). Node i maps to stripe i % kNumLockStripes.
    //
    // Granularity trade-off: one global lock would serialize all inserts;
    // a mutex per node costs 64 bytes each (~61 MiB at 1M nodes). A fixed
    // 4096-stripe array gives near-per-node parallelism for 256 KiB —
    // with 14 threads over 4096 stripes, false sharing of a stripe by
    // two threads working on different nodes is rare (~2%).
    //
    // Deadlock rule: never hold two stripe locks at once. Every critical
    // section locks one node, mutates it, and releases before touching
    // another. With at most one lock held per thread, no cycle can form.
    static constexpr size_t kNumLockStripes = 4096;
    mutable std::vector<std::mutex> node_locks_;

    std::mutex& lock_for(uint32_t node_id) const {
        return node_locks_[node_id % kNumLockStripes];
    }

    std::atomic<uint32_t> num_inserted_{0};

    uint32_t M0_;   // max connections at layer 0 = 2 * M
    float mL_;      // level multiplier = 1 / ln(M)

    std::mt19937 rng_;

    // Graph arena: neighbor slot arrays live here instead of per-list heap
    // allocations, so a node's neighbor IDs are contiguous (cache-friendly
    // traversal) and the whole graph frees at once. Chunked so it serves both
    // build() (first chunk sized exactly from the pre-drawn levels — one
    // chunk, no waste) and incremental insert/load (grow by adding chunks).
    // Allocation is single-threaded ONLY: build()'s serial pre-allocation
    // pass, or serial insert/load. Parallel workers never allocate — they only
    // mutate counts/slots under stripe locks — so the arena is never touched
    // concurrently.
    std::vector<std::unique_ptr<MemoryArena>> arena_chunks_;

    uint32_t layer_capacity(uint32_t layer) const {
        return layer == 0 ? M0_ : config_.M;
    }
    // Slots a node of the given level needs across all its layers.
    size_t block_slots_for_level(uint32_t level) const {
        return static_cast<size_t>(M0_) + static_cast<size_t>(level) * config_.M;
    }
    // Allocate n_slots uint32_t from the arena. Single-threaded use only.
    uint32_t* arena_alloc(size_t n_slots);
    // Allocate a node's per-layer neighbor lists (no-op if already allocated).
    void allocate_node(Node& node, uint32_t level);
};

} // namespace lattice
