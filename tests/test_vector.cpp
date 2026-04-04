#include "vector.h"
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>

// ── Generation tests ───────────────────────────────────────────────────────

TEST(VectorDataset, GenerateCreatesCorrectSize) {
    auto ds = lattice::generate_random_vectors(100, 64);
    EXPECT_EQ(ds.num_vectors, 100);
    EXPECT_EQ(ds.dimension, 64);
    EXPECT_EQ(ds.data.size(), 100 * 64);
}

TEST(VectorDataset, GenerateValuesInRange) {
    auto ds = lattice::generate_random_vectors(500, 32);
    for (float val : ds.data) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LT(val, 1.0f);
    }
}

TEST(VectorDataset, GenerateDeterministicWithSeed) {
    auto ds1 = lattice::generate_random_vectors(50, 16, 123);
    auto ds2 = lattice::generate_random_vectors(50, 16, 123);
    EXPECT_EQ(ds1.data, ds2.data);
}

TEST(VectorDataset, GenerateDifferentSeeds) {
    auto ds1 = lattice::generate_random_vectors(50, 16, 1);
    auto ds2 = lattice::generate_random_vectors(50, 16, 2);
    EXPECT_NE(ds1.data, ds2.data);
}

// ── Accessor tests ─────────────────────────────────────────────────────────

TEST(VectorDataset, GetVectorReturnsCorrectPointer) {
    auto ds = lattice::generate_random_vectors(10, 4);

    // get_vector(i) should point to data[i * dimension]
    for (uint32_t i = 0; i < 10; ++i) {
        const float* vec = ds.get_vector(i);
        EXPECT_EQ(vec, ds.data.data() + i * 4);
    }
}

TEST(VectorDataset, GetVectorOutOfRangeThrows) {
    auto ds = lattice::generate_random_vectors(10, 4);
    EXPECT_THROW(ds.get_vector(10), std::out_of_range);
    EXPECT_THROW(ds.get_vector(100), std::out_of_range);
}

TEST(VectorDataset, GetVectorMutableWorks) {
    auto ds = lattice::generate_random_vectors(5, 3);
    float* vec = ds.get_vector(2);
    vec[0] = -1.0f;
    EXPECT_EQ(ds.data[2 * 3], -1.0f);
}

// ── Save/Load round-trip tests ─────────────────────────────────────────────

class VectorFileTest : public ::testing::Test {
protected:
    std::string test_path = "test_roundtrip.bin";

    void TearDown() override {
        std::filesystem::remove(test_path);
    }
};

TEST_F(VectorFileTest, SaveLoadRoundTrip) {
    auto original = lattice::generate_random_vectors(200, 128);
    lattice::save_vectors(test_path, original);
    auto loaded = lattice::load_vectors(test_path);

    EXPECT_EQ(loaded.num_vectors, original.num_vectors);
    EXPECT_EQ(loaded.dimension, original.dimension);
    EXPECT_EQ(loaded.data, original.data);
}

TEST_F(VectorFileTest, SaveLoadSmallDataset) {
    auto original = lattice::generate_random_vectors(1, 1);
    lattice::save_vectors(test_path, original);
    auto loaded = lattice::load_vectors(test_path);

    EXPECT_EQ(loaded.num_vectors, 1);
    EXPECT_EQ(loaded.dimension, 1);
    EXPECT_EQ(loaded.data.size(), 1);
    EXPECT_EQ(loaded.data[0], original.data[0]);
}

TEST_F(VectorFileTest, SaveLoadPreservesFileSize) {
    auto ds = lattice::generate_random_vectors(50, 32);
    lattice::save_vectors(test_path, ds);

    size_t expected = 8 + static_cast<size_t>(50) * 32 * sizeof(float);
    EXPECT_EQ(std::filesystem::file_size(test_path), expected);
}

TEST_F(VectorFileTest, LoadNonexistentFileThrows) {
    EXPECT_THROW(lattice::load_vectors("nonexistent.bin"), std::runtime_error);
}

// ── Edge cases ─────────────────────────────────────────────────────────────

TEST(VectorDataset, EmptyDataset) {
    auto ds = lattice::generate_random_vectors(0, 128);
    EXPECT_EQ(ds.num_vectors, 0);
    EXPECT_EQ(ds.dimension, 128);
    EXPECT_TRUE(ds.data.empty());
}
