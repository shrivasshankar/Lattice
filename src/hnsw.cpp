#include "hnsw.h"
#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <queue>
#include <stdexcept>

namespace lattice {

// ── Constructor ────────────────────────────────────────────────────────────

HNSWIndex::HNSWIndex(const VectorDataset& dataset, HNSWConfig config)
    : dataset_(dataset)
    , config_(config)
    , nodes_(dataset.num_vectors)
    , M0_(config.M * 2)
    , mL_(1.0f / std::log(static_cast<float>(config.M)))
    , node_locks_(kNumLockStripes)
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
    // dist returns [0, 1); map to (0, 1] because -log(0) is +inf, and
    // casting an infinite float to uint32_t is undefined behavior.
    float r = 1.0f - dist(rng_);
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
    insert(vector_id, random_level());
}

void HNSWIndex::insert(uint32_t vector_id, uint32_t node_level) {
    Node& node = nodes_[vector_id];
    node.level = node_level;
    node.inserted = true;
    node.neighbors.resize(node_level + 1);

    // First node is a special case — no graph to search yet. The check
    // and the claim happen in one critical section (check-and-act, not
    // check-then-act): with a plain "count == 0" test, two concurrent
    // first inserts could both see an empty index, and the loser would
    // return without wiring any edges — an unreachable orphan.
    {
        std::lock_guard<std::mutex> lock(entry_mutex_);
        if (!has_entry_) {
            entry_point_ = vector_id;
            max_layer_ = node_level;
            has_entry_ = true;
            num_inserted_.fetch_add(1);
            return;
        }
    }

    const float* query = dataset_.get_vector(vector_id);

    // Snapshot the entry pair once, under the lock, then work from locals.
    // Re-reading max_layer_ mid-descent could observe a concurrent update
    // and break the pair's invariant.
    uint32_t ep;
    uint32_t top_layer;
    {
        std::lock_guard<std::mutex> lock(entry_mutex_);
        ep = entry_point_;
        top_layer = max_layer_;
    }

    // Phase 1: descend through layers above the new node's level.
    // We're just finding a good starting point — no connections made here.
    for (int layer = static_cast<int>(top_layer);
         layer > static_cast<int>(node_level); --layer) {
        auto results = search_layer(query, {ep}, 1, layer);
        if (!results.empty()) {
            ep = results[0].index;
        }
    }

    // Phase 2: at each layer where the new node exists, find neighbors
    // and create bidirectional connections.
    uint32_t insert_from = std::min(node_level, top_layer);

    for (int layer = static_cast<int>(insert_from); layer >= 0; --layer) {
        auto candidates = search_layer(query, {ep}, config_.ef_construction, layer);

        uint32_t max_conn = (layer == 0) ? M0_ : config_.M;
        auto neighbors = select_neighbors(candidates, max_conn);

        // Forward connections: new node → neighbors. Locked because the
        // node becomes visible to searchers as soon as the first reverse
        // edge below publishes it; writes to its other layers would then
        // race with readers. Writing forward edges before reverse edges
        // also guarantees a reader that discovers this node sees its
        // outgoing list for this layer fully populated.
        {
            std::lock_guard<std::mutex> lock(lock_for(vector_id));
            node.neighbors[layer] = neighbors;
        }

        // Reverse connections: each neighbor → new node, taking exactly
        // one stripe lock at a time (see deadlock rule in hnsw.h). The
        // prune happens under the same lock as the push_back: releasing
        // in between would let a concurrent insert add an edge that our
        // recomputed list silently drops (lost update).
        for (uint32_t neighbor_id : neighbors) {
            std::lock_guard<std::mutex> lock(lock_for(neighbor_id));
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

    // If the new node is the tallest, it becomes the global entry point.
    // The check must happen under the lock: comparing against a snapshot
    // would let two concurrent inserts both "win" and the shorter one
    // could overwrite the taller one's promotion.
    {
        std::lock_guard<std::mutex> lock(entry_mutex_);
        if (node_level > max_layer_) {
            entry_point_ = vector_id;
            max_layer_ = node_level;
        }
    }

    num_inserted_.fetch_add(1);
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
    if (num_inserted_.load() == 0) return {};
    if (ef_search < k) ef_search = k;

    // Snapshot the entry pair (see insert for why)
    uint32_t ep;
    uint32_t top_layer;
    {
        std::lock_guard<std::mutex> lock(entry_mutex_);
        ep = entry_point_;
        top_layer = max_layer_;
    }

    // Phase 1: greedily descend from top layer to layer 1
    for (int layer = static_cast<int>(top_layer); layer >= 1; --layer) {
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
    const uint32_t n = dataset_.num_vectors;

    // Draw every node's level before inserting anything. Level assignment
    // is the only randomness in construction, so pulling it out of the
    // insert loop keeps levels deterministic (seed-reproducible) even once
    // inserts run concurrently and complete in nondeterministic order.
    std::vector<uint32_t> levels(n);
    for (uint32_t i = 0; i < n; ++i) {
        levels[i] = random_level();
    }

    // Serial path: the reference implementation. Bit-identical graphs,
    // used as the baseline for recall-parity tests and benchmarks.
    if (config_.num_threads == 1 || n < 2) {
        for (uint32_t i = 0; i < n; ++i) {
            insert(i, levels[i]);
        }
        return;
    }

    // Parallel path. Insert the first node serially so the entry point
    // exists before workers start; otherwise all workers race through the
    // first-insert path against an empty graph.
    insert(0, levels[0]);

    // Work distribution: one long-running task per worker, all claiming
    // node IDs from a shared atomic cursor. Compared to submitting one
    // task per insert this has no per-insert queue/future overhead, and
    // compared to static range partitioning it self-balances — workers
    // that draw expensive inserts simply claim fewer IDs.
    ThreadPool pool(config_.num_threads);
    std::atomic<uint32_t> next{1};

    std::vector<std::future<void>> workers;
    workers.reserve(pool.num_threads());
    for (uint32_t t = 0; t < pool.num_threads(); ++t) {
        workers.push_back(pool.submit([this, &levels, &next, n] {
            while (true) {
                uint32_t i = next.fetch_add(1);
                if (i >= n) return;
                insert(i, levels[i]);
            }
        }));
    }

    // Barrier: build() returns only when every insert has completed.
    // get() also rethrows any exception that escaped a worker.
    for (auto& f : workers) {
        f.get();
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
    // Visited tracking via a per-thread epoch-stamped array instead of a
    // hash set. "Visited this search" means stamps[i] == epoch; starting a
    // fresh search is just ++epoch — an O(1) clear with no per-call
    // allocation and no hashing, replacing millions of unordered_set
    // build/teardown cycles across a full index build.
    //
    // thread_local because the parallel build runs search_layer on many
    // threads at once; each thread owning its own buffer is the
    // synchronization (no shared state, no lock).
    static thread_local std::vector<uint32_t> visited_stamps;
    static thread_local uint32_t visited_epoch = 0;
    const uint32_t num_vec = dataset_.num_vectors;
    if (visited_stamps.size() < num_vec) {
        visited_stamps.assign(num_vec, 0);
        visited_epoch = 0;
    }
    // Bump the epoch to invalidate all prior marks. On wraparound (after
    // ~4B searches) stale stamps could alias the new epoch, so clear once.
    if (++visited_epoch == 0) {
        std::fill(visited_stamps.begin(), visited_stamps.end(), 0);
        visited_epoch = 1;
    }

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
        if (visited_stamps[ep] == visited_epoch) continue;
        visited_stamps[ep] = visited_epoch;
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

        // Expand: visit all neighbors of this candidate.
        //
        // Snapshot the neighbor list under the node's stripe lock before
        // iterating. A concurrent insert may grow or prune this exact
        // vector; iterating the live vector across a reallocation is a
        // use-after-free. The copy may be momentarily stale (a just-added
        // edge missing), which only affects which candidates we expand —
        // never the validity of the graph or the results.
        std::vector<uint32_t> nbrs_snapshot;
        {
            std::lock_guard<std::mutex> lock(lock_for(best.index));
            const auto& nbrs = nodes_[best.index].neighbors;
            if (layer < nbrs.size()) {
                nbrs_snapshot = nbrs[layer];
            }
        }

        for (uint32_t neighbor_id : nbrs_snapshot) {
            if (visited_stamps[neighbor_id] == visited_epoch) continue;
            visited_stamps[neighbor_id] = visited_epoch;

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
// Two policies, chosen by HNSWConfig::use_diversity_heuristic:
//
//   closest-M (default): keep the max_neighbors candidates nearest the base
//   node. Fastest — essentially just a sort. Plateaus at high ef because
//   connections can cluster in one direction, leaving regions hard to reach.
//
//   diversity (HNSW paper, Algorithm 4): process candidates closest-first and
//   keep one only if it is closer to the base than to every already-selected
//   neighbor. This prunes candidates "behind" an existing neighbor (same
//   direction), so the chosen M point in diverse directions and the graph
//   stays navigable from all sides — lifting recall at high ef. Costs up to
//   ~|candidates| x max_neighbors extra distance evaluations per call.

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
    selected.reserve(max_neighbors);

    if (!config_.use_diversity_heuristic) {
        // Closest-M: take the nearest max_neighbors directly.
        uint32_t count = std::min(max_neighbors, static_cast<uint32_t>(sorted.size()));
        for (uint32_t i = 0; i < count; ++i) {
            selected.push_back(sorted[i].index);
        }
        return selected;
    }

    // Diversity heuristic. candidate.distance is the distance to the base node
    // (from search_layer); candidate-to-neighbor distances are computed here.
    for (const SearchResult& cand : sorted) {
        if (selected.size() >= max_neighbors) break;

        const float* cand_vec = dataset_.get_vector(cand.index);
        bool diverse = true;
        for (uint32_t r : selected) {
            float dist_to_r = config_.distance_fn(
                cand_vec, dataset_.get_vector(r), dataset_.dimension
            );
            // Closer to an already-selected neighbor than to the base node:
            // this direction is already covered, so skip.
            if (dist_to_r < cand.distance) {
                diverse = false;
                break;
            }
        }
        if (diverse) selected.push_back(cand.index);
    }

    // Backfill: if pruning left us short, add the closest unselected
    // candidates so connectivity never drops below what closest-M would give.
    if (selected.size() < max_neighbors) {
        for (const SearchResult& cand : sorted) {
            if (selected.size() >= max_neighbors) break;
            if (std::find(selected.begin(), selected.end(), cand.index) == selected.end()) {
                selected.push_back(cand.index);
            }
        }
    }

    return selected;
}

// ── Serialization ─────────────────────────────────────────────────────────
//
// Binary format (all values little-endian uint32_t unless noted):
//
//   [Header]
//     magic:           0x4C415454 ("LATT")
//     version:         1
//     num_vectors:     total nodes in the index
//     M:               max connections per layer
//     entry_point:     ID of the entry point node
//     max_layer:       highest layer in the graph
//     num_inserted:    number of inserted nodes
//
//   [Node table — one entry per node]
//     level:           highest layer this node appears in
//     For each layer 0..level:
//       num_neighbors:   neighbor count at this layer
//       neighbor_ids[]:  uint32_t array of neighbor IDs
//
// The vectors themselves are NOT saved — the caller must provide the same
// dataset when constructing the index before calling load().

static constexpr uint32_t LATTICE_MAGIC   = 0x4C415454; // "LATT"
static constexpr uint32_t LATTICE_VERSION = 1;

void HNSWIndex::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file for writing: " + filename);

    auto write_u32 = [&](uint32_t val) {
        file.write(reinterpret_cast<const char*>(&val), sizeof(val));
    };

    // Header
    write_u32(LATTICE_MAGIC);
    write_u32(LATTICE_VERSION);
    write_u32(dataset_.num_vectors);
    write_u32(config_.M);
    write_u32(entry_point_);
    write_u32(max_layer_);
    write_u32(num_inserted_.load());

    // Node table
    for (uint32_t i = 0; i < dataset_.num_vectors; ++i) {
        const Node& node = nodes_[i];
        write_u32(node.level);
        write_u32(node.inserted ? 1 : 0);

        if (!node.inserted) continue;

        for (uint32_t layer = 0; layer <= node.level; ++layer) {
            const auto& nbrs = node.neighbors[layer];
            write_u32(static_cast<uint32_t>(nbrs.size()));
            if (!nbrs.empty()) {
                file.write(reinterpret_cast<const char*>(nbrs.data()),
                           nbrs.size() * sizeof(uint32_t));
            }
        }
    }

    if (!file) throw std::runtime_error("Error writing index file: " + filename);
}

void HNSWIndex::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file for reading: " + filename);

    auto read_u32 = [&]() -> uint32_t {
        uint32_t val;
        file.read(reinterpret_cast<char*>(&val), sizeof(val));
        return val;
    };

    // Header
    uint32_t magic = read_u32();
    if (magic != LATTICE_MAGIC) {
        throw std::runtime_error("Invalid index file (bad magic number)");
    }

    uint32_t version = read_u32();
    if (version != LATTICE_VERSION) {
        throw std::runtime_error("Unsupported index version: " + std::to_string(version));
    }

    uint32_t num_vectors = read_u32();
    if (num_vectors != dataset_.num_vectors) {
        throw std::runtime_error("Index was built for " + std::to_string(num_vectors) +
                                 " vectors but dataset has " + std::to_string(dataset_.num_vectors));
    }

    uint32_t saved_M = read_u32();
    if (saved_M != config_.M) {
        throw std::runtime_error("Index M=" + std::to_string(saved_M) +
                                 " but config M=" + std::to_string(config_.M));
    }

    entry_point_  = read_u32();
    max_layer_    = read_u32();
    num_inserted_.store(read_u32());
    has_entry_ = (num_inserted_.load() > 0);

    // Node table
    nodes_.resize(num_vectors);
    for (uint32_t i = 0; i < num_vectors; ++i) {
        Node& node = nodes_[i];
        node.level = read_u32();
        node.inserted = (read_u32() == 1);

        if (!node.inserted) {
            node.neighbors.clear();
            continue;
        }

        node.neighbors.resize(node.level + 1);
        for (uint32_t layer = 0; layer <= node.level; ++layer) {
            uint32_t num_nbrs = read_u32();
            node.neighbors[layer].resize(num_nbrs);
            if (num_nbrs > 0) {
                file.read(reinterpret_cast<char*>(node.neighbors[layer].data()),
                          num_nbrs * sizeof(uint32_t));
            }
        }
    }

    if (!file) throw std::runtime_error("Error reading index file: " + filename);
}

// ── Getters ────────────────────────────────────────────────────────────────

const std::vector<uint32_t>& HNSWIndex::get_neighbors(uint32_t node_id, uint32_t layer) const {
    return nodes_[node_id].neighbors[layer];
}

uint32_t HNSWIndex::get_node_level(uint32_t node_id) const {
    return nodes_[node_id].level;
}

} // namespace lattice
