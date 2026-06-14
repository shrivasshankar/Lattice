# Lattice

A high-performance approximate nearest neighbor (ANN) vector search engine built from scratch in C++17. No external libraries for core algorithms — every component is hand-written: HNSW graph construction, SIMD-accelerated distance computation, custom memory allocator, and multi-threaded index construction and query processing.

## About this project

Started in February 2026 with reading (HNSW paper, FAISS source) and
prototyping in scratch files. The dense commit cadence in early April is
the shipping push of work that had been ongoing for ~2 months. Decisions
like the bump allocator, the NEON-vs-SSE split, and the parameter sweep
around M=32, efC=400 came out of that earlier exploration. The parallel
build (striped per-node locking, ThreadSanitizer-verified) was added in
June 2026.

## Performance

Benchmarked on Apple Silicon (M-series, 14 cores), 128-dimensional vectors
(uniform-random dataset; query sets noted per benchmark below), k=10, M=32, efC=400. The build-scaling, FAISS
head-to-head, and epoch-visited-list numbers were freshly measured against
the current build; the SIMD (7.4x/4.7x), arena (5.1x), and index-load (626x)
figures are from earlier work and were **not** re-measured this run. Build
times are single representative runs on one machine — not cross-machine
claims.

### Parallel build scaling (50K vectors)

`build()` distributes inserts across a thread pool with fine-grained
striped per-node locking. Speedup vs the single-threaded reference:

| threads | build time | speedup |
|---|---|---|
| 1  | 37.47s | 1.00x |
| 2  | 17.84s | 2.10x |
| 4  |  8.24s | 4.55x |
| 8  |  4.22s | 8.88x |
| 14 |  3.29s | **11.39x** |

Recall is unchanged across thread counts — the parallel build produces a
different graph each run (insert order is nondeterministic) but recall
stays at parity with the serial build, asserted in CI and verified
race-free under ThreadSanitizer. Scaling is slightly superlinear at 2–8
threads (cache effects on smaller per-thread working sets) and falls off
toward 14 as memory bandwidth saturates.

> Note on query sets: `scaling_benchmark` evaluates recall on uniform-random
> queries (~91% at ef=200), while the FAISS head-to-head below uses the
> exported *formula* queries (95.5% at ef=200). The two recall numbers for
> "50K, ef=200" differ because the query distributions differ — within each
> comparison both engines face byte-identical queries.

### SIFT1M — real-world benchmark (weight this one)

SIFT1M (1,000,000 128-dim image descriptors — the standard ANN benchmark)
with its provided ground truth. Both engines on identical data, 14 threads,
M=32, efC=400, recall@10:

| ef_search | Lattice recall | FAISS recall | Lattice latency | FAISS latency |
|---|---|---|---|---|
| 50   | 95.6% | **97.7%** | 0.236 ms | 0.016 ms |
| 100  | 97.9% | **99.5%** | 0.365 ms | 0.029 ms |
| 200  | 98.9% | **99.9%** | 0.660 ms | 0.053 ms |
| 500  | 99.3% | **99.9%** | 1.349 ms | 0.124 ms |
| 1000 | 99.4% | **99.9%** | 2.320 ms | 0.254 ms |

Build: **Lattice 111.6s, FAISS 80.7s** (FAISS ~1.4x faster).

On real data FAISS leads on all three axes — build (~1.4x), recall (~1–2
points), and latency (~12x) — from its mature, SIMD-heavy, years-tuned
implementation. Lattice's recall is genuinely competitive (95.6–99.4%, within
~1–2 points of FAISS across all ef), which is the result that matters for a
from-scratch HNSW. **Note this is the opposite of the synthetic build result
below:** on uniform-random vectors Lattice builds ~1.6x *faster*, but on real
SIFT1M FAISS builds ~1.4x faster. The build-speed comparison is
dataset-dependent — real data is the honest one to weight, and there Lattice
does not beat FAISS; it lands within ~1.4x.

### Synthetic-data comparison with FAISS (uniform-random)

These 50K/100K results use uniform-random vectors, an easier and less
representative distribution than SIFT1M. The build-speed win shown here is
specific to this synthetic data and **does not hold on real data** (see SIFT1M
above); it's kept for the scaling and recall detail. Both engines run on
byte-identical vectors and queries (dumped by `export_dataset`, loaded by
`tools/faiss_comparison.py`), M=32, efC=400, k=10, 200 queries, 14 cores.

**50K vectors** — build: **Lattice 3.4s, FAISS 5.5s** (Lattice ~1.6x faster, synthetic only)

| ef_search | Lattice recall@10 | FAISS recall@10 | Lattice latency | FAISS latency |
|---|---|---|---|---|
| 50   | **89.4%** | 77.0%  | 0.077 ms | 0.009 ms |
| 100  | **93.7%** | 91.6%  | 0.138 ms | 0.014 ms |
| 200  | **95.5%** | 95.0%  | 0.278 ms | 0.028 ms |
| 500  | 98.0%     | **100.0%** | 0.788 ms | 0.080 ms |
| 1000 | 100.0%    | 100.0% | 1.914 ms | 0.169 ms |

**100K vectors** — build: **Lattice 10.0s, FAISS 16.1s** (Lattice ~1.6x faster, synthetic only)

| ef_search | Lattice recall@10 | FAISS recall@10 | Lattice latency | FAISS latency |
|---|---|---|---|---|
| 50   | **70.8%** | 58.0%  | 0.079 ms | 0.014 ms |
| 100  | **81.4%** | 78.4%  | 0.158 ms | 0.017 ms |
| 200  | 92.6%     | 92.7%  | 0.359 ms | 0.032 ms |
| 500  | 97.0%     | **99.8%** | 1.216 ms | 0.096 ms |
| 1000 | 97.8%     | **100.0%** | 2.964 ms | 0.209 ms |

Honest reading of the numbers (this machine, this dataset — 128-dim
uniform-random vectors, faiss-cpu at default HNSW settings; not a general claim):

- **Build (dataset-dependent — do not over-read):** on these synthetic
  50K/100K sets Lattice builds ~1.6x faster (5.45s→3.38s at 50K), but on real
  SIFT1M FAISS builds ~1.4x faster (80.7s vs 111.6s). The synthetic build-win
  does **not** generalize. What's robust is that parallel construction plus the
  epoch visited-list took the original ~12–14x-slower serial build to within
  ~1.4x of FAISS on real data (and ahead on synthetic) — a real gain, just not
  "faster than FAISS" in general.
- **Recall, low ef:** Lattice reaches higher recall than FAISS at matched ef
  (50K ef=50: 89.4% vs 77.0%; ef=200: 95.5% vs 95.0%) — except they are
  effectively tied at 100K ef=200 (92.6% vs 92.7%). Caveat: ef is not iso-cost
  across engines. Lattice's higher recall comes with ~10x higher per-query
  latency, so at matched *latency* FAISS would run a larger ef and the
  comparison favors FAISS.
- **Recall, high ef (with the default closest-M selection):** at 100K Lattice
  plateaus around 97–98% while FAISS reaches ~100%; at 50K, FAISS leads at
  ef=500 (100.0% vs 98.0%) but the two tie at 100.0% by ef=1000. This plateau
  is the cost of closest-M neighbor selection. Enabling the **diversity
  heuristic** (`use_diversity_heuristic`, HNSW paper Algorithm 4) closes it —
  50K ef=500 reaches 100.0% and 100K ef=1000 reaches 100.0%, matching FAISS —
  for ~2.3x build cost. See the neighbor-selection table below.
- **Query latency:** FAISS is clearly faster per query — ~8.6–11x at 50K and
  ~6–14x at 100K (the gap grows with ef and dataset size at 100K; roughly flat
  across ef at 50K until ef=1000). This narrowed from ~28x before the
  visited-list optimization but remains FAISS's domain, driven by AVX2 SIMD
  (8 floats/instruction vs NEON's 4) and a more optimized query path.

### Neighbor selection: recall vs build time

Neighbor selection during build is a policy knob
(`HNSWConfig::use_diversity_heuristic`). Closest-M (default) keeps the nearest
M candidates — fastest build. The diversity heuristic keeps a candidate only
if it is closer to the node than to every already-chosen neighbor, so
connections spread in diverse directions and the graph stays navigable at high
ef. recall@10 (50K and 100K, identical data):

| metric | closest-M (default) | diversity |
|---|---|---|
| 50K build (14T) | **3.4s** | 8.4s |
| 50K recall @ ef=50 | **89.4%** | 84.5% |
| 50K recall @ ef=500 | 98.0% | **100.0%** |
| 100K build (14T) | **10.0s** | 21.7s |
| 100K recall @ ef=200 | 92.6% | **94.3%** |
| 100K recall @ ef=1000 | 97.8% | **100.0%** |

The trade is clean: closest-M is the faster build (on synthetic data ahead of
FAISS, on real SIFT1M within ~1.4x) but plateaus at high ef; diversity matches
FAISS recall at high ef but builds ~2.3x slower than closest-M and slightly
lowers low-ef recall. `tools/plot_pareto.py` renders the recall-vs-latency curve.

Reproduce:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target export_dataset scaling_benchmark
./build/export_dataset 50000 200 128
./build/export_dataset 100000 200 128
./build/scaling_benchmark 50000 128      # Lattice build scaling
python3 tools/faiss_comparison.py 50000 100000   # head-to-head (needs faiss-cpu)
```

### Component-Level Speedups

| Optimization | Speedup |
|---|---|
| Parallel build (14 cores, 50K) | **11.4x** vs single-thread |
| Epoch-stamped visited list (vs `unordered_set`) | **~2.0x** build (2.05x serial, up to 2.36x at 14T); ~2.8x query (50K, ef=200) |
| SIMD L2 distance (NEON) | **7.4x** vs scalar |
| SIMD cosine distance | **4.7x** vs scalar |
| Arena allocator (standalone microbenchmark) | **5.1x** vs malloc |
| Index load vs rebuild | **626x** faster |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Thread Pool                       │
│  submit() → [task queue] → worker threads (N cores)  │
│  drives both parallel build and parallel query       │
├─────────────────────────────────────────────────────┤
│                    HNSW Index                        │
│  Multi-layer graph with probabilistic skip-list      │
│  structure. Greedy descent + beam search at layer 0. │
│  Concurrent build: striped per-node locks (4096      │
│  mutexes), one lock held at a time (deadlock-free).  │
├──────────────────────┬──────────────────────────────┤
│   SIMD Distance      │     Memory Arena             │
│   ARM NEON / x86 SSE │     Bump allocator           │
│   4 floats/instr     │   backs graph neighbor       │
│   4x unrolled ILP    │   lists (fixed-capacity)     │
├──────────────────────┴──────────────────────────────┤
│                  Vector Storage                      │
│  Contiguous row-major float arrays + binary I/O      │
└─────────────────────────────────────────────────────┘
```

> Note: the HNSW graph's neighbor lists are allocated from the arena as
> fixed-capacity blocks (sized once from the pre-drawn levels, contiguous per
> node). The arena's all-allocate-then-free-at-destruction lifetime matches
> HNSW exactly, and the fixed-capacity blocks never reallocate — which also
> removed the use-after-free hazard that the concurrent reader had to guard
> against when neighbor lists were growable `std::vector`s.

## Concurrency

The parallel build is the result of a deliberate, incremental hardening of
the insert/search paths:

- **Striped locking** — 4096 mutexes shared across all nodes (`node i →
  stripe i % 4096`): near per-node parallelism at constant 256 KiB
  (4096 × 64 B) instead of ~61 MiB of per-node mutexes at 1M nodes
  (64 B per `std::mutex` on this toolchain; ~40 B on libstdc++/Linux).
- **One lock at a time** — every critical section holds a single stripe lock,
  making deadlock impossible by construction (no hold-and-wait).
- **Snapshot reads** — searches copy a neighbor list under its lock before
  iterating, so a concurrent insert reallocating that vector can't cause a
  use-after-free.
- **Invariant-protected globals** — atomic node counter; entry point and max
  layer guarded as a pair; first-insert and entry-point promotion done as
  check-and-act under a lock.
- **Verified** — recall-parity and no-orphan tests run in CI, and a
  ThreadSanitizer CI job builds with `-fsanitize=thread` and runs the
  concurrency tests on every push.

`HNSWConfig::num_threads` controls build parallelism (1 = serial reference,
0 = all hardware cores).

## Components

| File | What it does |
|---|---|
| `src/vector.h/cpp` | Contiguous vector storage, binary save/load |
| `src/distance.h/cpp` | Scalar L2 and cosine distance |
| `src/simd_distance.h/cpp` | SIMD-accelerated distance (NEON + SSE) |
| `src/search.h/cpp` | Brute-force KNN baseline, recall measurement |
| `src/hnsw.h/cpp` | HNSW index: parallel insert, build, search, save/load |
| `src/allocator.h/cpp` | Arena bump allocator + fixed-size pool allocator |
| `src/thread_pool.h/cpp` | Thread pool with task queue and futures |
| `tools/scaling_benchmark.cpp` | Build-time scaling across thread counts |
| `Dockerfile` | Multi-stage build: compile, test, minimal runtime image |
| `.github/workflows/ci.yml` | CI on Ubuntu (SSE) + macOS (NEON) + TSan + Docker |

## Key Design Decisions

- **Squared L2 distance** — skips `sqrt()` since it doesn't change nearest-neighbor ordering.
- **SIMD with 4 accumulators** — maximizes instruction-level parallelism by avoiding pipeline stalls from data dependencies.
- **Striped per-node locks** — fine-grained concurrency for the parallel build without the memory cost of a mutex per node; one lock held at a time keeps it deadlock-free.
- **Epoch-stamped visited list** — replaces a per-call `unordered_set` in the beam search; "visited" means `stamp[i] == epoch`, so clearing is an O(1) epoch bump with no allocation. Per-thread, so it needs no synchronization.
- **Arena-backed graph** — `src/allocator.h`'s bump-pointer arena (5.1x vs `malloc` in a standalone microbenchmark) backs the HNSW graph: each node's per-layer neighbor lists are fixed-capacity slot arrays carved from the arena, sized once from the pre-drawn levels and laid out contiguously per node. Because the blocks never reallocate, the concurrent search reads them under the stripe lock as a pure consistency snapshot (no use-after-free), and the whole graph frees at once.
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

103 tests covering correctness, edge cases, serialization, parallel-build
correctness, and performance benchmarks.

ThreadSanitizer build (race detection):

```bash
cmake -B build-tsan -DLATTICE_TSAN=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-tsan --target lattice_tests -j
./build-tsan/lattice_tests --gtest_filter='HNSWParallel.*'
```

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
- ThreadSanitizer build + concurrency tests
- Docker image build + benchmark run
