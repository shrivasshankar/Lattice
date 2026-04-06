#include "hnsw.h"

#include <algorithm>
#include <cmath>
#include <queue>

namespace lattice {

// ── Constructor ────────────────────────────────────────────────────────────

HNSWIndex::HNSWIndex(const VectorDataset& dataset, HNSWConfig config)
    : dataset_(dataset)
    , config_(config)
    , nodes_(dataset.num_vectors)
    , M0_(config.M * 2)
    , mL_(1.0f / std::log(static_cast<float>(config.M)))
    , rng_(config.seed)
{}

// ── Layer selection ────────────────────────────────────────────────────────
//
// Returns a random level drawn from a geometric-like distribution.
// Most nodes get level 0. Each additional level is exponentially rarer.
// This is what gives HNSW its "hierarchical" property — the top layers
// are sparse express lanes, the bottom layer is dense.

uint32_t HNSWIndex::random_level() {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    return static_cast<uint32_t>(-std::log(r) * mL_);
}

// ── Insert ─────────────────────────────────────────────────────────────────
//
// The HNSW insert algorithm:
//   1. Pick a random level for the new node
//   2. From the top layer down to (level+1), greedily find the closest
//      existing node — this is just navigating the "express lanes"
//   3. From min(level, max_layer) down to layer 0, find ef_construction
//      candidates and connect the new node to the M closest ones
//   4. If the new node's level is the highest yet, it becomes the entry point

void HNSWIndex::insert(uint32_t vector_id) {
    uint32_t node_level = random_level();

    Node& node = nodes_[vector_id];
    node.level = node_level;
    node.inserted = true;
    node.neighbors.resize(node_level + 1);

    // First node is a special case — no graph to search yet
    if (num_inserted_ == 0) {
        entry_point_ = vector_id;
        max_layer_ = node_level;
        num_inserted_++;
        return;
    }

    const float* query = dataset_.get_vector(vector_id);
    uint32_t ep = entry_point_;

    // Phase 1: descend through layers above the new node's level.
    // We're just finding a good starting point — no connections made here.
    for (int layer = static_cast<int>(max_layer_);
         layer > static_cast<int>(node_level); --layer) {
        auto results = search_layer(query, {ep}, 1, layer);
        if (!results.empty()) {
            ep = results[0].index;
        }
    }

    // Phase 2: at each layer where the new node exists, find neighbors
    // and create bidirectional connections.
    uint32_t insert_from = std::min(node_level, max_layer_);

    for (int layer = static_cast<int>(insert_from); layer >= 0; --layer) {
        auto candidates = search_layer(query, {ep}, config_.ef_construction, layer);

        uint32_t max_conn = (layer == 0) ? M0_ : config_.M;
        auto neighbors = select_neighbors(candidates, max_conn);

        // Forward connections: new node → neighbors
        node.neighbors[layer] = neighbors;

        // Reverse connections: each neighbor → new node
        for (uint32_t neighbor_id : neighbors) {
            Node& neighbor = nodes_[neighbor_id];
            neighbor.neighbors[layer].push_back(vector_id);

            // If a neighbor now has too many connections, prune it
            if (neighbor.neighbors[layer].size() > max_conn) {
                const float* n_vec = dataset_.get_vector(neighbor_id);
                std::vector<SearchResult> n_candidates;
                for (uint32_t n_neighbor : neighbor.neighbors[layer]) {
                    float d = config_.distance_fn(
                        n_vec, dataset_.get_vector(n_neighbor), dataset_.dimension
                    );
                    n_candidates.push_back({n_neighbor, d});
                }
                neighbor.neighbors[layer] = select_neighbors(n_candidates, max_conn);
            }
        }

        // Use the closest candidate as entry point for the next layer down
        if (!candidates.empty()) {
            ep = candidates[0].index;
        }
    }

    // If the new node is the tallest, it becomes the global entry point
    if (node_level > max_layer_) {
        entry_point_ = vector_id;
        max_layer_ = node_level;
    }

    num_inserted_++;
}

// ── Search ─────────────────────────────────────────────────────────────────
//
// The HNSW search algorithm mirrors insert's traversal:
//   1. Start at the entry point, top layer
//   2. Greedily descend through layers above 0 (ef=1 at each layer)
//   3. At layer 0, run search_layer with ef=ef_search
//   4. Return the top k results from that search
//
// The ef_search parameter is the key quality knob at query time:
//   - ef_search = k:   fast but low recall
//   - ef_search = 200: slow but high recall
//   - ef_search = 500: very slow, near-perfect recall

std::vector<SearchResult> HNSWIndex::search(
    const float* query,
    uint32_t k,
    uint32_t ef_search
) {
    if (num_inserted_ == 0) return {};
    if (ef_search < k) ef_search = k;

    uint32_t ep = entry_point_;

    // Phase 1: greedily descend from top layer to layer 1
    for (int layer = static_cast<int>(max_layer_); layer >= 1; --layer) {
        auto results = search_layer(query, {ep}, 1, layer);
        if (!results.empty()) {
            ep = results[0].index;
        }
    }

    // Phase 2: thorough search at layer 0 with ef_search candidates
    auto candidates = search_layer(query, {ep}, ef_search, 0);

    // Return only the top k results
    if (candidates.size() > k) {
        candidates.resize(k);
    }
    return candidates;
}

void HNSWIndex::build() {
    for (uint32_t i = 0; i < dataset_.num_vectors; ++i) {
        insert(i);
    }
}

// ── Search within a single layer ───────────────────────────────────────────
//
// This is the greedy beam search at the heart of HNSW.
// It maintains two heaps:
//   - candidates (min-heap): nodes to explore, closest first
//   - results (max-heap): the ef best nodes found so far, farthest on top
//
// At each step, we take the closest unexplored candidate and visit its
// neighbors. If a neighbor is closer than the worst result, it gets added
// to both heaps. We stop when the closest candidate is farther than the
// worst result — no more improvement is possible.

std::vector<SearchResult> HNSWIndex::search_layer(
    const float* query,
    const std::vector<uint32_t>& entry_ids,
    uint32_t ef,
    uint32_t layer
) {
    std::unordered_set<uint32_t> visited;

    // Min-heap: closest unvisited candidate on top
    auto cmp_min = [](const SearchResult& a, const SearchResult& b) {
        return a.distance > b.distance;
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp_min)>
        candidates(cmp_min);

    // Max-heap: farthest result on top (bounded to size ef)
    auto cmp_max = [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp_max)>
        results(cmp_max);

    // Seed with entry points
    for (uint32_t ep : entry_ids) {
        if (visited.count(ep)) continue;
        visited.insert(ep);
        float d = config_.distance_fn(dataset_.get_vector(ep), query, dataset_.dimension);
        candidates.push({ep, d});
        results.push({ep, d});
    }

    while (!candidates.empty()) {
        auto best = candidates.top();
        candidates.pop();

        // Termination: closest candidate is farther than worst result
        if (best.distance > results.top().distance && results.size() >= ef) {
            break;
        }

        // Expand: visit all neighbors of this candidate
        const auto& nbrs = nodes_[best.index].neighbors;
        if (layer < nbrs.size()) {
            for (uint32_t neighbor_id : nbrs[layer]) {
                if (visited.count(neighbor_id)) continue;
                visited.insert(neighbor_id);

                float d = config_.distance_fn(
                    dataset_.get_vector(neighbor_id), query, dataset_.dimension
                );

                if (results.size() < ef || d < results.top().distance) {
                    candidates.push({neighbor_id, d});
                    results.push({neighbor_id, d});
                    if (results.size() > ef) {
                        results.pop();
                    }
                }
            }
        }
    }

    // Convert to sorted vector (closest first)
    std::vector<SearchResult> result_vec;
    result_vec.reserve(results.size());
    while (!results.empty()) {
        result_vec.push_back(results.top());
        results.pop();
    }
    std::sort(result_vec.begin(), result_vec.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.distance < b.distance;
              });

    return result_vec;
}

// ── Neighbor selection ─────────────────────────────────────────────────────
//
// Simple strategy: pick the closest max_neighbors from the candidate set.
// The HNSW paper also describes a "heuristic" selection that prefers
// diverse neighbors (close to the node but far from each other). We'll
// keep it simple for now — this is sufficient for good recall.

std::vector<uint32_t> HNSWIndex::select_neighbors(
    const std::vector<SearchResult>& candidates,
    uint32_t max_neighbors
) {
    auto sorted = candidates;
    std::sort(sorted.begin(), sorted.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.distance < b.distance;
              });

    std::vector<uint32_t> selected;
    uint32_t count = std::min(max_neighbors, static_cast<uint32_t>(sorted.size()));
    selected.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        selected.push_back(sorted[i].index);
    }

    return selected;
}

// ── Getters ────────────────────────────────────────────────────────────────

const std::vector<uint32_t>& HNSWIndex::get_neighbors(uint32_t node_id, uint32_t layer) const {
    return nodes_[node_id].neighbors[layer];
}

uint32_t HNSWIndex::get_node_level(uint32_t node_id) const {
    return nodes_[node_id].level;
}

} // namespace lattice
