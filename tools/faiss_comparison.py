"""
Head-to-head comparison: Lattice HNSW vs Facebook FAISS HNSW.

To be a fair comparison, FAISS must run on the *exact same* vectors and queries
that Lattice benchmarks against. Lattice generates its data with C++'s
std::mt19937, which cannot be byte-reproduced from NumPy, so we don't try to
regenerate it here. Instead we load the binary files produced by the
`export_dataset` tool, which dumps the identical dataset/queries Lattice uses.

Workflow:
    cmake --build build --target export_dataset
    ./build/export_dataset 50000 200
    ./build/export_dataset 100000 200
    python3 tools/faiss_comparison.py

File format (matches lattice::save_vectors):
    uint32 count, uint32 dim, then count*dim little-endian float32.
"""
import numpy as np
import faiss
import time
import struct
import sys


def load_vectors(path):
    """Load a Lattice .bin vector file into an (count, dim) float32 array."""
    with open(path, "rb") as f:
        count, dim = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(count * dim * 4), dtype="<f4")
    return data.reshape(count, dim).copy()


def run_faiss_benchmark(N, dim, k, M, ef_construction, ef_search_values, num_queries=200):
    data = load_vectors(f"dataset_{N}.bin")
    queries = load_vectors(f"queries_{num_queries}.bin")
    assert data.shape == (N, dim), f"dataset shape {data.shape} != {(N, dim)}"
    assert queries.shape[1] == dim, f"query dim {queries.shape[1]} != {dim}"

    # Exact ground truth via brute force (same data as Lattice's brute_force_knn).
    index_flat = faiss.IndexFlatL2(dim)
    index_flat.add(data)
    _, gt_indices = index_flat.search(queries, k)

    # FAISS HNSW with the same parameters as Lattice (M=32, efC=400).
    index_hnsw = faiss.IndexHNSWFlat(dim, M)
    index_hnsw.hnsw.efConstruction = ef_construction

    t0 = time.time()
    index_hnsw.add(data)
    build_time = time.time() - t0

    print(f"\n=== FAISS HNSW (N={N}, dim={dim}, M={M}, efC={ef_construction}) ===")
    print(f"Dataset: same vectors as Lattice (loaded from dataset_{N}.bin)")
    print(f"Build time: {build_time:.1f}s\n")

    print(f"{'ef_search':>10} | {'recall@'+str(k):>10} | {'latency':>10}")
    print("-" * 38)

    for ef_s in ef_search_values:
        index_hnsw.hnsw.efSearch = ef_s

        t0 = time.time()
        _, indices = index_hnsw.search(queries, k)
        search_time = time.time() - t0
        avg_ms = (search_time / num_queries) * 1000

        total_recall = 0.0
        for q in range(num_queries):
            true_set = set(gt_indices[q])
            found = sum(1 for idx in indices[q] if idx in true_set)
            total_recall += found / k
        avg_recall = total_recall / num_queries * 100

        print(f"{ef_s:>10} | {avg_recall:>9.1f}% | {avg_ms:>8.3f} ms")

    return build_time


if __name__ == "__main__":
    print("=" * 60)
    print("  Lattice vs FAISS - Head-to-Head Comparison")
    print("=" * 60)

    sizes = [int(a) for a in sys.argv[1:]] or [50000, 100000]
    ef_searches = [50, 100, 200, 500, 1000]

    for N in sizes:
        run_faiss_benchmark(N, 128, 10, 32, 400, ef_searches)

    print("\nDone.")
