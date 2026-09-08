import csv
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from matplotlib.colors import LogNorm

# ---- Paths ----
WITHDRAWS_CSV = "data/processed_privacypools_data_withdraws.csv"
# CSVs produced by the knapsack scripts
DEP_COUNTER_PATH = "data/Heuristics/Knapsack/2-sum/knapsack_2_withdraws_pairs_per_deposit.csv"
PAIR_COUNTER_PATH = "data/Heuristics/Knapsack/2-sum/knapsack_2_withdraws_deposits_per_pair.csv"
# Output directory for the figures
FIG_DIR = "figures/Heuristics/Knapsack/2-sum/"

SHOW_PLOT = False  # True to display on screen in addition to saving


def decimal_fraction_hamming_weight(val) -> int:
    """Number of non-zero fractional digits (base-10 Hamming weight)."""
    d = Decimal(str(val)).normalize()
    parts = f"{d:f}".split(".")
    if len(parts) < 2:
        return 0
    return sum(1 for ch in parts[1].rstrip("0") if ch != "0")


# ================================================================
# 1. Read the CSVs
# ================================================================
dep_pairs_count = []
dep_amounts = []
dep_hw = []
with open(DEP_COUNTER_PATH, newline="") as f:
    for row in csv.DictReader(f):
        pc = int(row["pairs_count"])
        amt = float(row["dep_amount"])
        dep_pairs_count.append(pc)
        dep_amounts.append(amt)
        dep_hw.append(decimal_fraction_hamming_weight(amt))

dep_pairs_count = np.array(dep_pairs_count)
dep_amounts = np.array(dep_amounts)
dep_hw = np.array(dep_hw)

dep_amounts_with_pair = dep_amounts[dep_pairs_count > 0]

# per-pair + aggregation by withdraw
# occ_per_withdraw       : number of pairs (all) in which the withdraw appears
# occ_disc_per_withdraw  : number of DISCRIMINATING pairs (explaining a single
#                          deposit) in which the withdraw appears -> figure 5
pair_deposit_count = []
occ_per_withdraw = {}
occ_disc_per_withdraw = {}
with open(PAIR_COUNTER_PATH, newline="") as f:
    for row in csv.reader(f):
        if row[0] == "w1_id":
            continue
        w1_id, w2_id, cnt = int(row[0]), int(row[1]), int(row[2])
        pair_deposit_count.append(cnt)
        occ_per_withdraw[w1_id] = occ_per_withdraw.get(w1_id, 0) + 1
        occ_per_withdraw[w2_id] = occ_per_withdraw.get(w2_id, 0) + 1
        if cnt == 1:  # discriminating pair: explains a single deposit
            occ_disc_per_withdraw[w1_id] = occ_disc_per_withdraw.get(w1_id, 0) + 1
            occ_disc_per_withdraw[w2_id] = occ_disc_per_withdraw.get(w2_id, 0) + 1

pair_deposit_count = np.array(pair_deposit_count)

withdraw_amounts = []
with open(WITHDRAWS_CSV, newline="") as f:
    for row in csv.DictReader(f):
        withdraw_amounts.append(float(row["eth_amount"]))

withdraw_hw = [decimal_fraction_hamming_weight(a) for a in withdraw_amounts]
w_occ = np.array([occ_disc_per_withdraw.get(i, 0) for i in range(len(withdraw_amounts))])
w_hw = np.array(withdraw_hw)


# ================================================================
# Helper: scatter with density-colored points, symlog x-axis
#
#   - symlog: linear in [-0.5, 0.5] (shows the "0 column") then log beyond.
#     Allows displaying the == 0 counters without breaking the log scale.
#   - density: we count the points per (integer_x, integer_y) cell and color
#     each point by its cell density. No jitter on x (additive jitter distorts
#     positions on a log scale).
# ================================================================
def density_scatter(x, y, xlabel, ylabel, title, outpath, cmap):
    x = np.asarray(x); y = np.asarray(y)
    # density: number of points sharing the same integer cell (x, y)
    from collections import Counter
    cell = Counter(zip(x.tolist(), y.tolist()))
    dens = np.array([cell[(xi, yi)] for xi, yi in zip(x.tolist(), y.tolist())])

    # jitter ONLY on y (integer), none on x so as not to distort the log
    jitter_y = np.random.normal(0, 0.08, size=len(y))

    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(x, y + jitter_y, c=dens, cmap=cmap, norm=LogNorm(),
                    s=18, alpha=0.6, edgecolors="none")
    ax.set_xscale("symlog", linthresh=1)  # linear below 1 (0 column), log beyond
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_yticks(np.arange(0, int(y.max()) + 1, 1))
    ax.grid(True, linestyle="--", alpha=0.4)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Point density (count per integer cell)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)


# ================================================================
# FIGURE 1: histogram of the number of pairs per deposit (0 included)
#   symlog x-axis: narrow bin [0, 0.5) for the value 0, then log bins from 1
#   onward. The x=0 bar stays visible without crushing the rest.
# ================================================================
plt.figure(figsize=(10, 6))
max_pc = dep_pairs_count.max()
log_bins = np.logspace(0, np.log10(max_pc + 1), 60)
bins = np.concatenate(([0, 0.5], log_bins))  # narrow bin [0,0.5) for x==0
plt.hist(dep_pairs_count, bins=bins, color="#5b92e5",
         edgecolor="black", linewidth=0.5, linestyle=":")
plt.title("Figure 1 - Distribution of the number of withdrawal pairs per deposit (including 0)",
          fontsize=11)
plt.xlabel("Number of matching 2-withdrawal pairs per deposit (symlog)")
plt.ylabel("Frequency (number of deposits)")
plt.yscale("log")
plt.xscale("symlog", linthresh=1)  # linear below 1 (0 column), log beyond
plt.xlim(left=0)                   # cut the empty negative part of the symlog axis
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR + "knapsack_fig1_pairs_per_deposit_hist.png", dpi=150)

# ================================================================
# FIGURE 2: distribution of deposit amounts over the whole scale
#   Two series overlaid with transparency on a single plot:
#     - blue   : all kept deposits
#     - orange : deposits with at least one matching pair
#   Logarithmic bins + log/log axes: readable from 0.01 to ~48 ETH despite the
#   concentration below 1 ETH. The gap between the two series shows, at each
#   amount, the share of deposits WITHOUT any pair.
# ================================================================
plt.figure(figsize=(12, 6))
pos_amounts = dep_amounts[dep_amounts > 0]
lo, hi = pos_amounts.min(), pos_amounts.max()
bins = np.logspace(np.log10(lo), np.log10(hi), 70)

plt.hist(dep_amounts, bins=bins, color="#5b92e5", alpha=0.65,
         edgecolor="white", linewidth=0.3,
         label=f"All kept deposits (n={len(dep_amounts)})")
plt.hist(dep_amounts_with_pair, bins=bins, color="#f07853", alpha=0.75,
         edgecolor="white", linewidth=0.3,
         label=f"Deposits with >=1 matching pair (n={len(dep_amounts_with_pair)})")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Deposit amount (ETH, net)")
plt.ylabel("# Deposits")
plt.title("Figure 2 - Deposit amount distribution: all kept vs. with matching pair",
          fontsize=11)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR + "knapsack_fig2_deposit_amounts_hist.png", dpi=150)

# ================================================================
# FIGURE 3: histogram of the per-withdraw-pair counter
#   log x-axis + logarithmic bins (the counter starts at 1, so no log(0)
#   problem). Avoids the "comb" effect of integer bins.
# ================================================================
plt.figure(figsize=(10, 6))
max_pdc = pair_deposit_count.max()
bins = np.logspace(0, np.log10(max_pdc + 1), 60)
plt.hist(pair_deposit_count, bins=bins, color="#5b92e5", edgecolor="black", linewidth=0.5, linestyle=":")
plt.title("Figure 3 - Distribution of the number of deposits explained per withdrawal pair", fontsize=11)
plt.xlabel("Number of deposits matched by the same withdrawal pair")
plt.ylabel("Frequency (number of pairs)")
plt.yscale("log")
plt.xscale("log")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR + "knapsack_fig3_deposits_per_pair_hist.png", dpi=150)

# ================================================================
# FIGURE 4: density scatter  x = 2-sum link counter (deposit), y = HW deposit
# ================================================================
density_scatter(
    dep_pairs_count, dep_hw,
    xlabel="2-sum link counter (pairs per deposit, symlog)",
    ylabel="Hamming weight of deposit amount",
    title="Figure 4 - Deposit Hamming weight vs number of 2-sum links (color = density)",
    outpath=FIG_DIR + "knapsack_fig4_scatter_deposit_hw_vs_links.png",
    cmap="viridis",
)

# ================================================================
# FIGURE 5: density scatter  x = occurrences in discriminating pairs, y = HW withdraw
# ================================================================
density_scatter(
    w_occ, w_hw,
    xlabel="Occurrence counter in discriminating pairs (x=1 pairs only, symlog)",
    ylabel="Hamming weight of withdrawal amount",
    title="Figure 5 - Withdrawal Hamming weight vs occurrences in discriminating pairs (color = density)",
    outpath=FIG_DIR + "knapsack_fig5_scatter_withdraw_hw_vs_occurrences.png",
    cmap="viridis",
)

print("Figures saved to", FIG_DIR)
print(f"  deposits kept           : {len(dep_amounts)}")
print(f"  deposits with >=1 pair  : {len(dep_amounts_with_pair)}")
print(f"  distinct matching pairs : {len(pair_deposit_count)}")
print(f"  withdraws               : {len(withdraw_amounts)}")

if SHOW_PLOT:
    plt.show()