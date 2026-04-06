#pragma once

#include "vector.h"
#include "distance.h"
#include "search.h"

#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

namespace lattice {

struct HNSWConfig {
    uint32_t M = 16;                // max connections per node per layer
    uint32_t ef_construction = 200; // candidate list size during insert
    uint32_t seed = 42;
    DistanceFn distance_fn = l2_distance;
};

class HNSWIndex {
public:
    explicit HNSWIndex(const VectorDataset& dataset, HNSWConfig config = {});

    // Insert a single vector into the index by its ID in the dataset.
    void insert(uint32_t vector_id);

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

    // Getters for testing and inspection
    uint32_t get_max_layer() const { return max_layer_; }
    uint32_t get_entry_point() const { return entry_point_; }
    uint32_t num_nodes() const { return num_inserted_; }
    const std::vector<uint32_t>& get_neighbors(uint32_t node_id, uint32_t layer) const;
    uint32_t get_node_level(uint32_t node_id) const;

private:
    struct Node {
        std::vector<std::vector<uint32_t>> neighbors;  // neighbors[layer] = neighbor IDs
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
    uint32_t entry_point_ = 0;
    uint32_t max_layer_ = 0;
    uint32_t num_inserted_ = 0;

    uint32_t M0_;   // max connections at layer 0 = 2 * M
    float mL_;      // level multiplier = 1 / ln(M)

    std::mt19937 rng_;
};

} // namespace lattice
