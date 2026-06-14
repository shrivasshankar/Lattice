// SIFT1M / GIST1M benchmark harness.
//
// Loads the standard ANN-benchmark file formats (no download here — drop the
// dataset in and run):
//   base.fvecs        N base vectors    (per vector: int32 dim, then dim floats)
//   query.fvecs       Q query vectors   (same layout)
//   groundtruth.ivecs Q rows of true-k  (per row: int32 k, then k int32 ids)
//
// Recall is measured against the PROVIDED ground truth (no brute force needed),
// which is what makes a 1M-vector run practical. Build time uses the parallel
// path; flip --diversity for the high-recall neighbor-selection mode.
//
// Usage:
//   ./sift_benchmark <base.fvecs> <query.fvecs> <groundtruth.ivecs> [threads] [diversity]
//   ./sift_benchmark sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs 14 0

#include "hnsw.h"
#include "vector.h"
#include "search.h"
#include "simd_distance.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using namespace lattice;
using Clock = std::chrono::steady_clock;

// Read an .fvecs file into a VectorDataset. Dimension is read from the first
// record and every record is asserted to match it.
static VectorDataset load_fvecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    VectorDataset ds;
    ds.dimension = 0;
    ds.num_vectors = 0;

    int32_t dim = 0;
    while (f.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        if (ds.dimension == 0) ds.dimension = static_cast<uint32_t>(dim);
        if (static_cast<uint32_t>(dim) != ds.dimension)
            throw std::runtime_error("inconsistent dim in " + path);
        size_t base = ds.data.size();
        ds.data.resize(base + dim);
        f.read(reinterpret_cast<char*>(ds.data.data() + base), dim * sizeof(float));
        ds.num_vectors++;
    }
    return ds;
}

// Read an .ivecs ground-truth file: row i = true nearest-neighbor ids of query i.
static std::vector<std::vector<uint32_t>> load_ivecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    std::vector<std::vector<uint32_t>> rows;
    int32_t k = 0;
    while (f.read(reinterpret_cast<char*>(&k), sizeof(k))) {
        std::vector<uint32_t> row(k);
        f.read(reinterpret_cast<char*>(row.data()), k * sizeof(int32_t));
        rows.push_back(std::move(row));
    }
    return rows;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s <base.fvecs> <query.fvecs> <groundtruth.ivecs> [threads] [diversity]\n",
            argv[0]);
        return 1;
    }
    uint32_t threads   = (argc > 4) ? std::atoi(argv[4]) : 14;
    bool diversity     = (argc > 5) ? std::atoi(argv[5]) != 0 : false;

    auto base    = load_fvecs(argv[1]);
    auto queries = load_fvecs(argv[2]);
    auto truth   = load_ivecs(argv[3]);

    printf("base: %u x %u | queries: %u | threads: %u | selection: %s\n",
           base.num_vectors, base.dimension, queries.num_vectors, threads,
           diversity ? "diversity" : "closest-M");

    HNSWConfig cfg;
    cfg.distance_fn = l2_distance_simd;
    cfg.num_threads = threads;
    cfg.use_diversity_heuristic = diversity;

    HNSWIndex index(base, cfg);
    auto t0 = Clock::now();
    index.build();
    printf("build: %.2fs\n\n", std::chrono::duration<double>(Clock::now() - t0).count());

    const uint32_t k = 10;
    printf("%10s | %10s | %10s\n", "ef_search", "recall@10", "latency");
    printf("--------------------------------------\n");
    for (uint32_t ef : {50u, 100u, 200u, 500u, 1000u}) {
        auto s0 = Clock::now();
        double recall = 0.0;
        for (uint32_t q = 0; q < queries.num_vectors; ++q) {
            auto res = index.search(queries.get_vector(q), k, ef);
            std::unordered_set<uint32_t> gt(truth[q].begin(),
                                            truth[q].begin() + std::min<size_t>(k, truth[q].size()));
            uint32_t hit = 0;
            for (const auto& r : res) if (gt.count(r.index)) hit++;
            recall += static_cast<double>(hit) / k;
        }
        double ms = std::chrono::duration<double>(Clock::now() - s0).count()
                    / queries.num_vectors * 1000.0;
        printf("%10u | %9.1f%% | %7.3f ms\n", ef, recall / queries.num_vectors * 100.0, ms);
    }
    return 0;
}
