#include "hnsw.h"
#include "vector.h"
#include "distance.h"
#include "simd_distance.h"
#include "search.h"

#include <gtest/gtest.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>

class HNSWSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        dataset = lattice::generate_random_vectors(1000, 64, 42);
        index = std::make_unique<lattice::HNSWIndex>(
            dataset, lattice::HNSWConfig{.M = 16, .ef_construction = 100, .seed = 42});
        index->build();
    }

    void TearDown() override {
        std::remove("test_index.lattice");
    }

    lattice::VectorDataset dataset;
    std::unique_ptr<lattice::HNSWIndex> index;
};

TEST_F(HNSWSerializationTest, SaveDoesNotCrash) {
    EXPECT_NO_THROW(index->save("test_index.lattice"));
}

TEST_F(HNSWSerializationTest, LoadRestoresMetadata) {
    index->save("test_index.lattice");

    lattice::HNSWIndex loaded(dataset, {.M = 16, .ef_construction = 100, .seed = 42});
    loaded.load("test_index.lattice");

    EXPECT_EQ(loaded.num_nodes(), index->num_nodes());
    EXPECT_EQ(loaded.get_max_layer(), index->get_max_layer());
    EXPECT_EQ(loaded.get_entry_point(), index->get_entry_point());
}

TEST_F(HNSWSerializationTest, LoadRestoresGraphStructure) {
    index->save("test_index.lattice");

    lattice::HNSWIndex loaded(dataset, {.M = 16, .ef_construction = 100, .seed = 42});
    loaded.load("test_index.lattice");

    for (uint32_t i = 0; i < 1000; ++i) {
        EXPECT_EQ(loaded.get_node_level(i), index->get_node_level(i));
        for (uint32_t layer = 0; layer <= index->get_node_level(i); ++layer) {
            EXPECT_EQ(loaded.get_neighbors(i, layer), index->get_neighbors(i, layer));
        }
    }
}

TEST_F(HNSWSerializationTest, LoadedIndexSearchMatchesOriginal) {
    index->save("test_index.lattice");

    lattice::HNSWIndex loaded(dataset, {.M = 16, .ef_construction = 100, .seed = 42});
    loaded.load("test_index.lattice");

    // Run 50 search queries on both and compare results
    for (uint32_t q = 0; q < 50; ++q) {
        const float* query = dataset.get_vector(q);
        auto original_results = index->search(query, 10, 100);
        auto loaded_results = loaded.search(query, 10, 100);

        ASSERT_EQ(original_results.size(), loaded_results.size());
        for (size_t i = 0; i < original_results.size(); ++i) {
            EXPECT_EQ(original_results[i].index, loaded_results[i].index);
            EXPECT_FLOAT_EQ(original_results[i].distance, loaded_results[i].distance);
        }
    }
}

TEST_F(HNSWSerializationTest, LoadNonexistentFileThrows) {
    lattice::HNSWIndex loaded(dataset, {.M = 16});
    EXPECT_THROW(loaded.load("nonexistent_file.lattice"), std::runtime_error);
}

TEST_F(HNSWSerializationTest, LoadBadMagicThrows) {
    // Write garbage to a file
    std::ofstream f("test_index.lattice", std::ios::binary);
    uint32_t garbage = 0xDEADBEEF;
    f.write(reinterpret_cast<const char*>(&garbage), sizeof(garbage));
    f.close();

    lattice::HNSWIndex loaded(dataset, {.M = 16});
    EXPECT_THROW(loaded.load("test_index.lattice"), std::runtime_error);
}

TEST_F(HNSWSerializationTest, LoadMismatchedMThrows) {
    index->save("test_index.lattice");

    lattice::HNSWIndex loaded(dataset, {.M = 32}); // different M
    EXPECT_THROW(loaded.load("test_index.lattice"), std::runtime_error);
}

TEST_F(HNSWSerializationTest, LoadFasterThanBuild) {
    index->save("test_index.lattice");

    auto build_start = std::chrono::high_resolution_clock::now();
    lattice::HNSWIndex built(dataset, {.M = 16, .ef_construction = 100, .seed = 42});
    built.build();
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();

    auto load_start = std::chrono::high_resolution_clock::now();
    lattice::HNSWIndex loaded(dataset, {.M = 16, .ef_construction = 100, .seed = 42});
    loaded.load("test_index.lattice");
    auto load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

    std::cout << "  Build: " << std::fixed << std::setprecision(1) << build_ms << " ms\n"
              << "  Load:  " << load_ms << " ms\n"
              << "  Speedup: " << std::setprecision(0) << build_ms / load_ms << "x\n";

    EXPECT_LT(load_ms, build_ms) << "Loading should be much faster than building";
}
