"""
Parallel-build scaling: HNSW index construction speedup vs thread count
(50K vectors, Apple Silicon 14 cores, M=32, efC=400). Numbers match the
"Parallel build scaling" table in the README.

Usage: python3 tools/plot_build_scaling.py   # writes build_scaling.png
Requires matplotlib (pip install matplotlib).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THREADS = [1, 2, 4, 8, 14]
SPEEDUP = [1.00, 2.10, 4.55, 8.88, 11.39]
BUILD_S = [37.47, 17.84, 8.24, 4.22, 3.29]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(THREADS, THREADS, linestyle="--", color="#999999", linewidth=1.5, label="ideal (linear)")
ax.plot(THREADS, SPEEDUP, marker="o", color="#378ADD", linewidth=2, markersize=8, label="Lattice (measured)")

for t, s, b in zip(THREADS, SPEEDUP, BUILD_S):
    ax.annotate(f"{s:.1f}× / {b:.1f}s", (t, s), textcoords="offset points", xytext=(8, -4), fontsize=8)

ax.set_xlabel("threads")
ax.set_ylabel("build speedup vs single-thread")
ax.set_title("Parallel HNSW build scaling — 50K vectors, 14 cores")
ax.set_xticks(THREADS)
ax.set_ylim(0, 15)
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig("build_scaling.png", dpi=140)
print("wrote build_scaling.png")
