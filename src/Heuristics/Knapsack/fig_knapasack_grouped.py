import csv
import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----
DEP_1SUM = "data/Heuristics/Knapsack/1-sum/knapsack_1_withdraw_matches_per_deposit.csv"
DEP_2SUM = "data/Heuristics/Knapsack/2-sum/knapsack_2_withdraws_pairs_per_deposit.csv"
DEP_3SUM = "data/Heuristics/Knapsack/3-sum/knapsack_3_withdraws_triplets_per_deposit.csv"

OBJ_1SUM = "data/Heuristics/Knapsack/1-sum/knapsack_1_withdraw_deposits_per_withdraw.csv"
OBJ_2SUM = "data/Heuristics/Knapsack/2-sum/knapsack_2_withdraws_deposits_per_pair.csv"
OBJ_3SUM = "data/Heuristics/Knapsack/3-sum/knapsack_3_withdraws_deposits_per_triplet.csv"

FIG_DIR = "figures/Heuristics/Knapsack/grouped_sum/"

SHOW_PLOT = False

# Colors per variant
COLORS = {"1-Sum": "#5b92e5", "2-Sum": "#f0603a", "3-Sum": "#f0b028"}


def load_counter(path, colname):
    vals = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            vals.append(int(row[colname]))
    return np.array(vals)


def load_obj_counter(path):
    vals = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = header.index("deposits_count")
        for row in reader:
            vals.append(int(row[idx]))
    return np.array(vals)


def log_buckets(max_val, include_zero):
    """
    Build buckets at EQUIDISTANT positions (discrete categories):
      [0], [1], [2-3], [4-7], [8-15], [16-31], ... (powers of 2 beyond 1)
    Returns (edges, labels) where edges[i] = (lo, hi) inclusive.
    The equal spacing allows side-by-side bars of the same width, and avoids
    the 0 bar or the 0->1 range taking up too much space.
    """
    buckets = []
    labels = []
    if include_zero:
        buckets.append((0, 0)); labels.append("0")
    buckets.append((1, 1)); labels.append("1")
    lo = 2
    while lo <= max_val:
        hi = lo * 2 - 1
        buckets.append((lo, hi))
        labels.append(f"{lo}-{hi}" if hi > lo else f"{lo}")
        lo = hi + 1
    return buckets, labels


def bucketize(arr, buckets):
    """Count how many values of arr fall into each bucket."""
    counts = np.zeros(len(buckets), dtype=int)
    for i, (lo, hi) in enumerate(buckets):
        counts[i] = np.sum((arr >= lo) & (arr <= hi))
    return counts


def grouped_bar_plot(series, buckets, labels, xlabel, ylabel, title, outpath):
    """series: list of (label, array). Side-by-side bars per bucket."""
    n_series = len(series)
    x = np.arange(len(buckets))          # equidistant positions
    total_width = 0.8
    bar_w = total_width / n_series

    fig, ax = plt.subplots(figsize=(13, 6))
    for k, (lab, arr) in enumerate(series):
        counts = bucketize(arr, buckets)
        offset = (k - (n_series - 1) / 2) * bar_w
        ax.bar(x + offset, counts, width=bar_w, color=COLORS[lab],
               edgecolor="black", linewidth=0.3, label=f"{lab} (n={len(arr)})")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)


# ================================================================
# GROUPED FIGURE 1: number of links per deposit (the three "figure 1")
# ================================================================
d1 = load_counter(DEP_1SUM, "matches_count")
d2 = load_counter(DEP_2SUM, "pairs_count")
d3 = load_counter(DEP_3SUM, "triplets_count")

max_all = max(d1.max(), d2.max(), d3.max())
buckets, labels = log_buckets(max_all, include_zero=True)

grouped_bar_plot(
    [("1-Sum", d1), ("2-Sum", d2), ("3-Sum", d3)],
    buckets, labels,
    xlabel="Number of links per deposit",
    ylabel="Frequency (number of deposits)",
    title="Grouped Figure 1 - Links per deposit across 1-sum, 2-sum and 3-sum",
    outpath=FIG_DIR + "grouped_fig1_links_per_deposit.png",
)

# ================================================================
# GROUPED FIGURE 3: number of deposits per object (the three "figure 3")
# ================================================================
o1 = load_obj_counter(OBJ_1SUM)
o2 = load_obj_counter(OBJ_2SUM)
o3 = load_obj_counter(OBJ_3SUM)

max_obj = max(o1.max(), o2.max(), o3.max())
buckets_o, labels_o = log_buckets(max_obj, include_zero=False)  # counters >= 1

grouped_bar_plot(
    [("1-Sum", o1), ("2-Sum", o2), ("3-Sum", o3)],
    buckets_o, labels_o,
    xlabel="Number of deposits explained by the same object (withdrawal / pair / triplet)",
    ylabel="Frequency (number of objects)",
    title="Grouped Figure 3 - Deposits explained per object across 1-sum, 2-sum and 3-sum",
    outpath=FIG_DIR + "grouped_fig3_deposits_per_object.png",
)

print("Grouped figures saved to", FIG_DIR)
print(f"  1-sum deposits: {len(d1)} | 2-sum: {len(d2)} | 3-sum: {len(d3)}")
print(f"  1-sum objects : {len(o1)} | 2-sum: {len(o2)} | 3-sum: {len(o3)}")

if SHOW_PLOT:
    plt.show()