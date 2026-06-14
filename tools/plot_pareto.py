"""
Pareto plot: recall@10 vs query latency for Lattice (both neighbor-selection
modes) and FAISS, on 50K identical vectors. Up-and-to-the-left is better.

Measured this session (Apple Silicon, 14 cores, 128-dim, M=32, efC=400, k=10,
200 formula queries), ef_search in {50, 100, 200, 500, 1000}.

Usage: python3 tools/plot_pareto.py   # writes pareto_50k.png
Requires matplotlib (pip install matplotlib).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (latency_ms, recall_pct) per ef, low ef first.
LATTICE_CLOSEST_M = [(0.077, 89.4), (0.138, 93.7), (0.278, 95.5), (0.788, 98.0), (1.914, 100.0)]
LATTICE_DIVERSITY = [(0.080, 84.5), (0.145, 93.8), (0.299, 95.8), (0.968, 100.0), (2.167, 100.0)]
FAISS             = [(0.009, 77.0), (0.014, 91.6), (0.028, 95.0), (0.080, 100.0), (0.169, 100.0)]


def unzip(series):
    return [p[0] for p in series], [p[1] for p in series]


fig, ax = plt.subplots(figsize=(7, 4.5))
for series, label, color, marker, ls in [
    (LATTICE_CLOSEST_M, "Lattice closest-M (default)", "#378ADD", "o", "-"),
    (LATTICE_DIVERSITY, "Lattice diversity",           "#0F6E56", "^", "--"),
    (FAISS,             "FAISS",                        "#D85A30", "s", ":"),
]:
    x, y = unzip(series)
    ax.plot(x, y, marker=marker, linestyle=ls, color=color, label=label, linewidth=2, markersize=7)

ax.set_xscale("log")
ax.set_xlabel("query latency (ms, log scale) — lower is better")
ax.set_ylabel("recall@10 (%) — higher is better")
ax.set_title("Recall vs latency @ 50K vectors (identical data & queries)")
ax.set_ylim(70, 101)
ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("pareto_50k.png", dpi=140)
print("wrote pareto_50k.png")
