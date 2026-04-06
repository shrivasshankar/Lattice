#include "hnsw.h"
#include "vector.h"
#include "distance.h"
#include <gtest/gtest.h>
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
