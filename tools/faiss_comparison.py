"""
Head-to-head comparison: Lattice HNSW vs Facebook FAISS HNSW.
Same dataset, same parameters, same queries.
"""
import numpy as np
import faiss
import time
import subprocess
import struct
import sys

def run_faiss_benchmark(N, dim, k, M, ef_construction, ef_search_values, num_queries=200):
    np.random.seed(42)
    data = np.random.uniform(-1, 1, (N, dim)).astype('float32')

    # Queries (same generation as Lattice's main.cpp)
    queries = np.zeros((num_queries, dim), dtype='float32')
    for q in range(num_queries):
        for d in range(dim):
            queries[q, d] = float(q * dim + d) / (num_queries * dim)

    # Ground truth: brute force
    index_flat = faiss.IndexFlatL2(dim)
    index_flat.add(data)
    gt_distances, gt_indices = index_flat.search(queries, k)

    # FAISS HNSW
    index_hnsw = faiss.IndexHNSWFlat(dim, M)
    index_hnsw.hnsw.efConstruction = ef_construction

    t0 = time.time()
    index_hnsw.add(data)
    build_time = time.time() - t0

    print(f"\n=== FAISS HNSW (N={N}, dim={dim}, M={M}, efC={ef_construction}) ===")
    print(f"Build time: {build_time:.1f}s\n")

    print(f"{'ef_search':>10} | {'recall@'+str(k):>10} | {'latency':>10}")
    print("-" * 38)

    for ef_s in ef_search_values:
        index_hnsw.hnsw.efSearch = ef_s

        t0 = time.time()
        distances, indices = index_hnsw.search(queries, k)
        search_time = time.time() - t0
        avg_ms = (search_time / num_queries) * 1000

        # Compute recall
        total_recall = 0
        for q in range(num_queries):
            true_set = set(gt_indices[q])
            found = sum(1 for idx in indices[q] if idx in true_set)
            total_recall += found / k

        avg_recall = total_recall / num_queries * 100

        print(f"{ef_s:>10} | {avg_recall:>9.1f}% | {avg_ms:>8.3f} ms")

    return build_time


if __name__ == "__main__":
    print("=" * 60)
    print("  Lattice vs FAISS — Head-to-Head Comparison")
    print("=" * 60)

    for N in [50000, 100000, 500000]:
        ef_searches = [50, 100, 200, 500]

        # M=32, ef_construction=400 (matches Lattice's tuned config)
        run_faiss_benchmark(N, 128, 10, 32, 400, ef_searches)

    print("\nDone.")
