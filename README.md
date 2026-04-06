# Lattice

A high-performance approximate nearest neighbor (ANN) vector search engine built from scratch in C++17. No external libraries for core algorithms — every component is hand-written: HNSW graph construction, SIMD-accelerated distance computation, custom memory allocator, and multi-threaded query processing.

## Performance

Benchmarked on Apple Silicon (M-series), 128 dimensions, k=10:

### Recall vs Latency (50K vectors, M=32, efC=400)

| ef_search | recall@10 | latency |
|---|---|---|
| 50 | 85.8% | 0.185 ms |
| 100 | 91.8% | 0.369 ms |
| 200 | **95.5%** | 0.747 ms |
| 500 | **98.5%** | 1.791 ms |

### Scale Benchmarks

| Dataset | Build time | recall@200 | recall@500 | Throughput (14 threads) |
|---|---|---|---|---|
| 50K vectors | 64s | 95.5% | 98.5% | 8,846 qps |
| 100K vectors | 178s | 92.6% | 97.0% | 9,156 qps |
| 500K vectors | 2.6 hrs | 75.0% | 85.9% | 8,470 qps |

### Comparison with Facebook FAISS

Same dataset, same parameters (M=32, efC=400), same queries:

| Metric (50K, ef=200) | Lattice | FAISS |
|---|---|---|
| Recall@10 | **95.5%** | 97.0% |
| Query latency | 0.747 ms | 0.031 ms |
| Build time | 64s | 5.5s |

| Metric (100K, ef=500) | Lattice | FAISS |
|---|---|---|
| Recall@10 | **97.0%** | 93.9% |
| Query latency | 2.071 ms | 0.125 ms |

Lattice achieves competitive recall with FAISS — and higher recall at 100K vectors at the same ef setting. FAISS is faster due to AVX2 SIMD (8 floats/instruction vs NEON's 4), multi-threaded index build, and years of production optimization by Facebook's research team.

### Component-Level Speedups

| Optimization | Speedup |
|---|---|
| SIMD L2 distance (NEON) | **7.4x** vs scalar |
| SIMD cosine distance | **4.7x** vs scalar |
| Arena allocator | **5.1x** vs malloc |
| Multi-threaded search (14 cores) | **4.9x** vs single-thread |
| Index load vs rebuild | **626x** faster |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Thread Pool                       │
│  submit() → [task queue] → worker threads (N cores)  │
├─────────────────────────────────────────────────────┤
│                    HNSW Index                        │
│  Multi-layer graph with probabilistic skip-list      │
│  structure. Greedy descent + beam search at layer 0. │
├──────────────────────┬──────────────────────────────┤
│   SIMD Distance      │     Memory Arena             │
│   ARM NEON / x86 SSE │     Bump allocator           │
│   4 floats/instr     │     5x faster than malloc    │
│   4x unrolled ILP    │     Zero fragmentation       │
├──────────────────────┴──────────────────────────────┤
│                  Vector Storage                      │
│  Contiguous row-major float arrays + binary I/O      │
└─────────────────────────────────────────────────────┘
```

## Components

| File | What it does |
|---|---|
| `src/vector.h/cpp` | Contiguous vector storage, binary save/load |
| `src/distance.h/cpp` | Scalar L2 and cosine distance |
| `src/simd_distance.h/cpp` | SIMD-accelerated distance (NEON + SSE) |
| `src/search.h/cpp` | Brute-force KNN baseline, recall measurement |
| `src/hnsw.h/cpp` | HNSW index: insert, build, search, save/load |
| `src/allocator.h/cpp` | Arena bump allocator + fixed-size pool allocator |
| `src/thread_pool.h/cpp` | Thread pool with task queue and futures |
| `Dockerfile` | Multi-stage build: compile, test, minimal runtime image |
| `.github/workflows/ci.yml` | CI on Ubuntu (SSE) + macOS (NEON) + Docker |

## Key Design Decisions

- **Squared L2 distance** — skips `sqrt()` since it doesn't change nearest-neighbor ordering.
- **SIMD with 4 accumulators** — maximizes instruction-level parallelism by avoiding pipeline stalls from data dependencies.
- **Arena allocator for graph nodes** — HNSW allocates everything during build and frees everything at destruction. Arena semantics match perfectly.
- **`DistanceFn` as function pointer** — swap between scalar/SIMD/cosine at construction time with zero runtime overhead.
- **`ef_search` parameter** — single knob to trade recall for latency at query time.
- **Binary serialization** — save/load index graph in a compact binary format with magic number validation and version checking. Load is 626x faster than rebuild.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Test

```bash
cd build && ctest --output-on-failure
```

99 tests covering correctness, edge cases, serialization, and performance benchmarks.

## Run

```bash
./build/lattice
```

## Docker

```bash
docker build -t lattice:latest .
docker run --rm lattice:latest
```

## CI

GitHub Actions runs on every push and PR:
- Build + test on Ubuntu (x86/SSE) and macOS (ARM/NEON)
- Docker image build + benchmark run
