# Lattice

A high-performance approximate nearest neighbor (ANN) vector search engine built from scratch in C++17. No external libraries for core algorithms — every component is hand-written: HNSW graph construction, SIMD-accelerated distance computation, custom memory allocator, and multi-threaded query processing.

## Performance

Benchmarked on Apple Silicon (M-series), 50,000 vectors, 128 dimensions:

| Metric | Result |
|---|---|
| Index build time | 23.0s |
| Index save | 7.7 ms |
| Index load | 0.1 ms (626x faster than build) |
| Query throughput (1 thread) | 5,617 queries/sec |
| Query throughput (14 threads) | 26,716 queries/sec |
| Recall@10 (ef=100) | 79.2% |
| Query latency (ef=100) | 0.183 ms per query |

### Distance computation (1M operations, dim=128)

| Function | Scalar | SIMD (NEON) | Speedup |
|---|---|---|---|
| L2 distance | 47.0 ms | 6.3 ms | **7.4x** |
| Cosine distance | 75.6 ms | 16.0 ms | **4.7x** |

### Memory allocator (100K allocations × 64 bytes)

| Allocator | Time | Speedup |
|---|---|---|
| `malloc` | 1,647 μs | baseline |
| Arena | 320 μs | **5.1x** |

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
| `src/hnsw.h/cpp` | HNSW index: insert, build, search |
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

## Recall vs Latency Tradeoff

```
ef_search | recall@10 | avg latency
----------|-----------|------------
       10 |    38.6%  |   0.048 ms
       50 |    68.9%  |   0.164 ms
      100 |    79.3%  |   0.304 ms
      200 |    86.9%  |   0.582 ms
      500 |    89.5%  |   1.327 ms
```

Higher `ef_search` examines more candidates during search, improving recall at the cost of latency.
