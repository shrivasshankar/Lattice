# Lattice

A high-performance approximate nearest neighbor (ANN) vector search engine built from scratch in C++17. No external libraries for core algorithms — every component is hand-written: HNSW graph construction, SIMD-accelerated distance computation, custom memory allocator, and multi-threaded query processing.

## About this project

Started in February 2026 with reading (HNSW paper, FAISS source) and
prototyping in scratch files / [MadStorageV1, if relevant]. The dense
commit cadence in early April is the shipping push of work that had
been ongoing for ~2 months. Decisions like the bump allocator, the
NEON-vs-SSE split, and the parameter sweep around M=32, efC=400 came
out of that earlier exploration.

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

Both engines run on the **exact same vectors and queries** — Lattice's dataset is
dumped to disk by the `export_dataset` tool and loaded by `tools/faiss_comparison.py`,
so this is a true apples-to-apples comparison (not two independently generated datasets).
Same parameters throughout (M=32, efC=400, k=10, 200 queries).

**50K vectors** — build: Lattice 65.7s, FAISS 5.7s

| ef_search | Lattice recall@10 | FAISS recall@10 | Lattice latency | FAISS latency |
|---|---|---|---|---|
| 50 | **85.8%** | 75.9% | 0.189 ms | 0.014 ms |
| 100 | **91.8%** | 90.7% | 0.395 ms | 0.018 ms |
| 200 | **95.5%** | 95.0% | 0.786 ms | 0.027 ms |
| 500 | 98.5% | **100.0%** | 1.898 ms | 0.077 ms |
| 1000 | 100.0% | 100.0% | 3.498 ms | 0.169 ms |

**100K vectors** — build: Lattice 209.6s, FAISS 15.7s

| ef_search | Lattice recall@10 | FAISS recall@10 | Lattice latency | FAISS latency |
|---|---|---|---|---|
| 50 | **71.6%** | 58.5% | 0.185 ms | 0.010 ms |
| 100 | **82.2%** | 73.4% | 0.377 ms | 0.020 ms |
| 200 | **92.6%** | 91.9% | 0.789 ms | 0.033 ms |
| 500 | 97.0% | **99.8%** | 2.460 ms | 0.096 ms |
| 1000 | 97.8% | **100.0%** | 3.506 ms | 0.206 ms |

Honest reading of the numbers:

- **Low-to-mid ef:** Lattice reaches *higher* recall than FAISS at the same ef (e.g. 50K ef=200: 95.5% vs 95.0%; 100K ef=50: 71.6% vs 58.5%). Lattice's candidate list converges quickly on this data.
- **High ef:** FAISS pulls ahead to ~100% while Lattice plateaus (98.5% at 50K, 97.0% at 100K). FAISS uses the HNSW *diversity* neighbor-selection heuristic; Lattice uses simple closest-M selection, which caps peak recall (see Key Design Decisions).
- **Speed:** FAISS is ~10-25x lower query latency and ~12x faster to build — AVX2 SIMD (8 floats/instruction vs NEON's 4), multi-threaded index construction, and years of production optimization.

Reproduce:

```bash
cmake --build build --target export_dataset
./build/export_dataset 50000 200
./build/export_dataset 100000 200
python3 tools/faiss_comparison.py 50000 100000
```

### Component-Level Speedups

| Optimization | Speedup |
|---|---|
| SIMD L2 distance (NEON) | **7.4x** vs scalar |
| SIMD cosine distance | **4.7x** vs scalar |
| Arena allocator (standalone microbenchmark) | **5.1x** vs malloc |
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

> Note: the Memory Arena is a standalone, tested component. The HNSW index does
> not currently allocate its graph through the arena — it uses `std::vector`.
> Wiring the arena into the graph is a planned optimization, not yet implemented.

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
- **Arena/pool allocators (standalone)** — `src/allocator.h` implements a bump-pointer arena and a fixed-size pool allocator, benchmarked at 5.1x vs `malloc` for many small allocations. They are *not yet wired into the HNSW graph* (which currently stores nodes/neighbor lists in `std::vector`); HNSW's all-allocate-then-free-at-destruction lifetime is an ideal future fit for the arena.
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
