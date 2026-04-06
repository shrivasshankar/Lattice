#include "hnsw.h"
#include "vector.h"
#include "distance.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <cmath>

// ── Basic construction tests ───────────────────────────────────────────────

TEST(HNSWInsert, BuildDoesNotCrash) {
    auto ds = lattice::generate_random_vectors(500, 32, 42);
    lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 50, .seed = 42});
    index.build();

    EXPECT_EQ(index.num_nodes(), 500);
}

TEST(HNSWInsert, SingleNodeInsert) {
    auto ds = lattice::generate_random_vectors(1, 8, 42);
    lattice::HNSWIndex index(ds);
    index.insert(0);

    EXPECT_EQ(index.num_nodes(), 1);
    EXPECT_EQ(index.get_entry_point(), 0);
}

TEST(HNSWInsert, EntryPointIsSet) {
    auto ds = lattice::generate_random_vectors(100, 16, 42);
    lattice::HNSWIndex index(ds, {.M = 8, .ef_construction = 50, .seed = 42});
    index.build();

    uint32_t ep = index.get_entry_point();
    EXPECT_LT(ep, 100);
    // Entry point should be at the max layer
    EXPECT_EQ(index.get_node_level(ep), index.get_max_layer());
}

TEST(HNSWInsert, AllNodesInserted) {
    auto ds = lattice::generate_random_vectors(200, 16, 99);
    lattice::HNSWIndex index(ds, {.M = 8, .ef_construction = 50, .seed = 99});
    index.build();

    EXPECT_EQ(index.num_nodes(), 200);
}

// ── Layer structure tests ──────────────────────────────────────────────────

TEST(HNSWInsert, MostNodesAtLayerZero) {
    auto ds = lattice::generate_random_vectors(1000, 32, 42);
    lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 50, .seed = 42});
    index.build();

    // Count nodes at each level
    uint32_t layer0_count = 0;
    uint32_t higher_count = 0;
    for (uint32_t i = 0; i < 1000; ++i) {
        if (index.get_node_level(i) == 0) layer0_count++;
        else higher_count++;
    }

    // Most nodes should be at level 0 (roughly 1 - 1/M fraction)
    EXPECT_GT(layer0_count, higher_count);
}

TEST(HNSWInsert, MaxLayerIsReasonable) {
    auto ds = lattice::generate_random_vectors(1000, 32, 42);
    lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 50, .seed = 42});
    index.build();

    // For 1000 nodes with M=16, max layer should typically be 2-5
    EXPECT_GE(index.get_max_layer(), 1);
    EXPECT_LE(index.get_max_layer(), 10);
}

// ── Graph connectivity tests ───────────────────────────────────────────────

TEST(HNSWInsert, NodesHaveNeighborsAtLayerZero) {
    auto ds = lattice::generate_random_vectors(100, 16, 42);
    lattice::HNSWIndex index(ds, {.M = 8, .ef_construction = 50, .seed = 42});
    index.build();

    // Every node (except possibly the very first) should have at least
    // one neighbor at layer 0
    uint32_t nodes_with_neighbors = 0;
    for (uint32_t i = 0; i < 100; ++i) {
        if (!index.get_neighbors(i, 0).empty()) {
            nodes_with_neighbors++;
        }
    }
    EXPECT_GE(nodes_with_neighbors, 99);
}

TEST(HNSWInsert, NeighborCountWithinLimits) {
    uint32_t M = 8;
    uint32_t M0 = M * 2;
    auto ds = lattice::generate_random_vectors(500, 16, 42);
    lattice::HNSWIndex index(ds, {.M = M, .ef_construction = 100, .seed = 42});
    index.build();

    for (uint32_t i = 0; i < 500; ++i) {
        uint32_t level = index.get_node_level(i);
        for (uint32_t layer = 0; layer <= level; ++layer) {
            uint32_t max_conn = (layer == 0) ? M0 : M;
            EXPECT_LE(index.get_neighbors(i, layer).size(), max_conn)
                << "Node " << i << " at layer " << layer
                << " has too many neighbors";
        }
    }
}

TEST(HNSWInsert, NeighborsAreBidirectional) {
    auto ds = lattice::generate_random_vectors(100, 8, 42);
    lattice::HNSWIndex index(ds, {.M = 8, .ef_construction = 50, .seed = 42});
    index.build();

    // For each node at layer 0, check that if A has B as a neighbor,
    // B's neighbor pruning may have removed A — so we check that MOST
    // connections are bidirectional, not all.
    uint32_t total_connections = 0;
    uint32_t bidirectional = 0;

    for (uint32_t i = 0; i < 100; ++i) {
        const auto& neighbors = index.get_neighbors(i, 0);
        for (uint32_t n : neighbors) {
            total_connections++;
            const auto& reverse = index.get_neighbors(n, 0);
            if (std::find(reverse.begin(), reverse.end(), i) != reverse.end()) {
                bidirectional++;
            }
        }
    }

    // At least 50% of connections should be bidirectional
    // (pruning can break some reverse connections)
    EXPECT_GT(bidirectional, total_connections / 2);
}

// ── Graph connectivity (BFS reachability) ──────────────────────────────────

TEST(HNSWStructure, AllNodesReachableFromEntryPoint) {
    auto ds = lattice::generate_random_vectors(500, 16, 42);
    lattice::HNSWIndex index(ds, {.M = 8, .ef_construction = 100, .seed = 42});
    index.build();

    // BFS from entry point at layer 0 — every node should be reachable
    std::unordered_set<uint32_t> visited;
    std::queue<uint32_t> frontier;
    frontier.push(index.get_entry_point());
    visited.insert(index.get_entry_point());

    while (!frontier.empty()) {
        uint32_t current = frontier.front();
        frontier.pop();
        for (uint32_t neighbor : index.get_neighbors(current, 0)) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                frontier.push(neighbor);
            }
        }
    }

    // Allow at most 1 unreachable node (the very first inserted node can
    // become orphaned if all its reverse connections get pruned)
    EXPECT_GE(visited.size(), 499)
        << "Only " << visited.size() << " of 500 nodes reachable from entry point";
}

TEST(HNSWStructure, HigherLayersReachable) {
    auto ds = lattice::generate_random_vectors(1000, 32, 42);
    lattice::HNSWIndex index(ds, {.M = 16, .ef_construction = 50, .seed = 42});
    index.build();

    // At every layer, all nodes in that layer should be reachable from the entry point
    for (uint32_t layer = 1; layer <= index.get_max_layer(); ++layer) {
        // Find all nodes at this layer
        std::vector<uint32_t> layer_nodes;
        for (uint32_t i = 0; i < 1000; ++i) {
            if (index.get_node_level(i) >= layer) {
                layer_nodes.push_back(i);
            }
        }
        if (layer_nodes.empty()) continue;

        // BFS from entry point at this layer
        std::unordered_set<uint32_t> visited;
        std::queue<uint32_t> frontier;
        frontier.push(index.get_entry_point());
        visited.insert(index.get_entry_point());

        while (!frontier.empty()) {
            uint32_t current = frontier.front();
            frontier.pop();
            if (index.get_node_level(current) >= layer) {
                for (uint32_t n : index.get_neighbors(current, layer)) {
                    if (visited.find(n) == visited.end()) {
                        visited.insert(n);
                        frontier.push(n);
                    }
                }
            }
        }

        // Count how many layer nodes we reached
        uint32_t reached = 0;
        for (uint32_t n : layer_nodes) {
            if (visited.count(n)) reached++;
        }
        EXPECT_EQ(reached, layer_nodes.size())
            << "Layer " << layer << ": only " << reached << " of "
            << layer_nodes.size() << " nodes reachable";
    }
}

// ── Neighbor quality tests ─────────────────────────────────────────────────

TEST(HNSWStructure, NeighborsAreActuallyClose) {
    auto ds = lattice::generate_random_vectors(200, 16, 42);
    lattice::HNSWIndex index(ds, {.M = 8, .ef_construction = 100, .seed = 42});
    index.build();

    // For a sample of nodes, check that their layer-0 neighbors are among
    // the closest vectors (not random far-away nodes).
    uint32_t good_neighbors = 0;
    uint32_t total_checked = 0;

    for (uint32_t i = 0; i < 50; ++i) {
        const float* vec_i = ds.get_vector(i);
        const auto& neighbors = index.get_neighbors(i, 0);

        // Compute distance to all other vectors, find the true 2*M closest
        std::vector<lattice::SearchResult> all_dists;
        for (uint32_t j = 0; j < 200; ++j) {
            if (j == i) continue;
            float d = lattice::l2_distance(vec_i, ds.get_vector(j), ds.dimension);
            all_dists.push_back({j, d});
        }
        std::sort(all_dists.begin(), all_dists.end(),
                  [](const lattice::SearchResult& a, const lattice::SearchResult& b) {
                      return a.distance < b.distance;
                  });

        // The true 4*M closest (generous bound)
        uint32_t check_range = std::min(static_cast<uint32_t>(all_dists.size()), 8u * 8u);
        std::unordered_set<uint32_t> close_set;
        for (uint32_t c = 0; c < check_range; ++c) {
            close_set.insert(all_dists[c].index);
        }

        // Most neighbors should be in the close set
        for (uint32_t n : neighbors) {
            total_checked++;
            if (close_set.count(n)) good_neighbors++;
        }
    }

    // At least 70% of neighbors should be among the true-closest vectors
    float quality = static_cast<float>(good_neighbors) / total_checked;
    EXPECT_GT(quality, 0.7f)
        << "Neighbor quality: " << quality << " (expected > 0.7)";
}

// ── Different M values ─────────────────────────────────────────────────────

TEST(HNSWStructure, WorksWithSmallM) {
    auto ds = lattice::generate_random_vectors(100, 8, 42);
    lattice::HNSWIndex index(ds, {.M = 4, .ef_construction = 50, .seed = 42});
    index.build();
    EXPECT_EQ(index.num_nodes(), 100);

    // All nodes should still have neighbors
    for (uint32_t i = 0; i < 100; ++i) {
        if (i != index.get_entry_point() || index.num_nodes() > 1) {
            EXPECT_FALSE(index.get_neighbors(i, 0).empty())
                << "Node " << i << " has no neighbors with M=4";
        }
    }
}

TEST(HNSWStructure, WorksWithLargeM) {
    auto ds = lattice::generate_random_vectors(100, 8, 42);
    lattice::HNSWIndex index(ds, {.M = 32, .ef_construction = 100, .seed = 42});
    index.build();
    EXPECT_EQ(index.num_nodes(), 100);

    // With M=32, nodes can have up to 64 neighbors at layer 0.
    // Most should have close to M neighbors (bounded by available nodes).
    for (uint32_t i = 0; i < 100; ++i) {
        EXPECT_LE(index.get_neighbors(i, 0).size(), 64);
    }
}

// ── Edge cases ─────────────────────────────────────────────────────────────

TEST(HNSWStructure, TwoNodeIndex) {
    auto ds = lattice::generate_random_vectors(2, 4, 42);
    lattice::HNSWIndex index(ds, {.M = 8, .ef_construction = 50, .seed = 42});
    index.build();

    EXPECT_EQ(index.num_nodes(), 2);

    // Both nodes should be neighbors of each other at layer 0
    const auto& n0 = index.get_neighbors(0, 0);
    const auto& n1 = index.get_neighbors(1, 0);

    bool zero_has_one = std::find(n0.begin(), n0.end(), 1) != n0.end();
    bool one_has_zero = std::find(n1.begin(), n1.end(), 0) != n1.end();

    EXPECT_TRUE(zero_has_one || one_has_zero);
}

TEST(HNSWStructure, NoSelfLoops) {
    auto ds = lattice::generate_random_vectors(200, 16, 42);
    lattice::HNSWIndex index(ds, {.M = 8, .ef_construction = 50, .seed = 42});
    index.build();

    for (uint32_t i = 0; i < 200; ++i) {
        uint32_t level = index.get_node_level(i);
        for (uint32_t layer = 0; layer <= level; ++layer) {
            const auto& neighbors = index.get_neighbors(i, layer);
            EXPECT_TRUE(std::find(neighbors.begin(), neighbors.end(), i) == neighbors.end())
                << "Node " << i << " has a self-loop at layer " << layer;
        }
    }
}

TEST(HNSWStructure, NoDuplicateNeighbors) {
    auto ds = lattice::generate_random_vectors(200, 16, 42);
    lattice::HNSWIndex index(ds, {.M = 8, .ef_construction = 50, .seed = 42});
    index.build();

    for (uint32_t i = 0; i < 200; ++i) {
        uint32_t level = index.get_node_level(i);
        for (uint32_t layer = 0; layer <= level; ++layer) {
            const auto& neighbors = index.get_neighbors(i, layer);
            std::unordered_set<uint32_t> unique(neighbors.begin(), neighbors.end());
            EXPECT_EQ(unique.size(), neighbors.size())
                << "Node " << i << " has duplicate neighbors at layer " << layer;
        }
    }
}

// ── Determinism test ───────────────────────────────────────────────────────

TEST(HNSWInsert, DeterministicWithSameSeed) {
    auto ds = lattice::generate_random_vectors(100, 16, 42);

    lattice::HNSWIndex idx1(ds, {.M = 8, .ef_construction = 50, .seed = 123});
    idx1.build();

    lattice::HNSWIndex idx2(ds, {.M = 8, .ef_construction = 50, .seed = 123});
    idx2.build();

    EXPECT_EQ(idx1.get_entry_point(), idx2.get_entry_point());
    EXPECT_EQ(idx1.get_max_layer(), idx2.get_max_layer());

    for (uint32_t i = 0; i < 100; ++i) {
        EXPECT_EQ(idx1.get_node_level(i), idx2.get_node_level(i));
        EXPECT_EQ(idx1.get_neighbors(i, 0), idx2.get_neighbors(i, 0));
    }
}
