import csv
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from matplotlib.colors import LogNorm

# ---- Paths ----
WITHDRAWS_CSV = "data/processed_privacypools_data_withdraws.csv"
# CSVs produced by get_knapsack_1_iteration_dict.py
DEP_COUNTER_PATH = "data/Heuristics/Knapsack/1-sum/knapsack_1_withdraw_matches_per_deposit.csv"
W_COUNTER_PATH = "data/Heuristics/Knapsack/1-sum/knapsack_1_withdraw_deposits_per_withdraw.csv"
FIG_DIR = "figures/Heuristics/Knapsack/1-sum/"

SHOW_PLOT = False
 
 
def decimal_fraction_hamming_weight(val) -> int:
    d = Decimal(str(val)).normalize()
    parts = f"{d:f}".split(".")
    if len(parts) < 2:
        return 0
    return sum(1 for ch in parts[1].rstrip("0") if ch != "0")
 
 
# ================================================================
# 1. Read the CSVs
# ================================================================
dep_matches_count = []
dep_amounts = []
dep_hw = []
with open(DEP_COUNTER_PATH, newline="") as f:
    for row in csv.DictReader(f):
        dep_matches_count.append(int(row["matches_count"]))
        amt = float(row["dep_amount"])
        dep_amounts.append(amt)
        dep_hw.append(decimal_fraction_hamming_weight(amt))
 
dep_matches_count = np.array(dep_matches_count)
dep_amounts = np.array(dep_amounts)
dep_hw = np.array(dep_hw)
 
dep_amounts_with_match = dep_amounts[dep_matches_count > 0]
 
# per-withdraw: number of deposits a withdraw explains (indexed by w_id)
w_deposit_count = {}
deposits_per_withdraw_vals = []
with open(W_COUNTER_PATH, newline="") as f:
    for row in csv.reader(f):
        if row[0] == "w_id":
            continue
        w_id, cnt = int(row[0]), int(row[1])
        w_deposit_count[w_id] = cnt
        deposits_per_withdraw_vals.append(cnt)
 
deposits_per_withdraw_vals = np.array(deposits_per_withdraw_vals)
 
# withdraws: amount + hw. w_id = aggregation order, but the per_withdraw CSV
# only indexes the matching withdraws. For figure 5 we re-read the amounts from
# the raw CSV and rebuild the aggregation by tx_hash (same order as the main
# script) in order to map w_id -> amount/hw.
raw_by_hash = {}
order = []
with open(WITHDRAWS_CSV, newline="") as f:
    for row in csv.DictReader(f):
        h = row["tx_hash"]
        amt = Decimal(str(row["eth_amount"]))
        if h in raw_by_hash:
            raw_by_hash[h] += amt
        else:
            raw_by_hash[h] = amt
            order.append(h)
 
# w_id assigned in order of first appearance (like the main script)
w_amount_by_id = {}
w_hw_by_id = {}
for w_id, h in enumerate(order):
    a = float(raw_by_hash[h])
    w_amount_by_id[w_id] = a
    w_hw_by_id[w_id] = decimal_fraction_hamming_weight(a)
 
n_w = len(order)
# for figure 5: occurrences (number of deposits) and HW, for ALL withdraws
# (0 if the withdraw explains no deposit)
w_occ = np.array([w_deposit_count.get(i, 0) for i in range(n_w)])
w_hw = np.array([w_hw_by_id[i] for i in range(n_w)])
 
 
# ================================================================
# Helper: density scatter, symlog x-axis
# ================================================================
def density_scatter(x, y, xlabel, ylabel, title, outpath, cmap):
    x = np.asarray(x); y = np.asarray(y)
    from collections import Counter
    cell = Counter(zip(x.tolist(), y.tolist()))
    dens = np.array([cell[(xi, yi)] for xi, yi in zip(x.tolist(), y.tolist())])
    jitter_y = np.random.normal(0, 0.08, size=len(y))
 
    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(x, y + jitter_y, c=dens, cmap=cmap, norm=LogNorm(),
                    s=18, alpha=0.6, edgecolors="none")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(left=0)
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
# FIGURE 1: histogram of the number of matches per deposit (0 included)
# ================================================================
plt.figure(figsize=(10, 6))
max_mc = dep_matches_count.max()
log_bins = np.logspace(0, np.log10(max_mc + 1), 60)
bins = np.concatenate(([0, 0.5], log_bins))
plt.hist(dep_matches_count, bins=bins, color="#5b92e5",
         edgecolor="black", linewidth=0.5, linestyle=":")
plt.title("Figure 1 (1-sum) - Number of matching withdrawals per deposit (including 0)",
          fontsize=11)
plt.xlabel("Number of matching withdrawals per deposit (symlog)")
plt.ylabel("Frequency (number of deposits)")
plt.yscale("log")
plt.xscale("symlog", linthresh=1)
plt.xlim(left=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR + "knapsack1_fig1_matches_per_deposit_hist.png", dpi=150)
 
# ================================================================
# FIGURE 2: distribution of deposit amounts (all vs with match)
# ================================================================
plt.figure(figsize=(12, 6))
# At HW=0, some amounts can be 0: np.log10(0) = -inf breaks the log bins.
# So we clamp the minimum to the smallest STRICTLY positive amount.
pos_amounts = dep_amounts[dep_amounts > 0]
lo, hi = pos_amounts.min(), pos_amounts.max()
bins = np.logspace(np.log10(lo), np.log10(hi), 70)
plt.hist(dep_amounts, bins=bins, color="#5b92e5", alpha=0.65,
         edgecolor="white", linewidth=0.3,
         label=f"All kept deposits (n={len(dep_amounts)})")
plt.hist(dep_amounts_with_match, bins=bins, color="#f07853", alpha=0.75,
         edgecolor="white", linewidth=0.3,
         label=f"Deposits with >=1 matching withdrawal (n={len(dep_amounts_with_match)})")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Deposit amount (ETH, net)")
plt.ylabel("# Deposits")
plt.title("Figure 2 (1-sum) - Deposit amount distribution: all kept vs. with matching withdrawal",
          fontsize=11)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR + "knapsack1_fig2_deposit_amounts_hist.png", dpi=150)
 
# ================================================================
# FIGURE 3: histogram of the per-withdraw counter
#   (number of deposits a single withdraw explains)
# ================================================================
plt.figure(figsize=(10, 6))
max_wdc = deposits_per_withdraw_vals.max()
bins = np.logspace(0, np.log10(max_wdc + 1), 60)
plt.hist(deposits_per_withdraw_vals, bins=bins, color="#5b92e5",
         edgecolor="black", linewidth=0.5, linestyle=":")
plt.title("Figure 3 (1-sum) - Number of deposits explained per withdrawal",
          fontsize=11)
plt.xlabel("Number of deposits matched by the same withdrawal")
plt.ylabel("Frequency (number of withdrawals)")
plt.yscale("log"); plt.xscale("log")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR + "knapsack1_fig3_deposits_per_withdraw_hist.png", dpi=150)
 
# ================================================================
# FIGURE 4: density scatter  x = matches counter (deposit), y = HW deposit
# ================================================================
density_scatter(
    dep_matches_count, dep_hw,
    xlabel="1-sum match counter (withdrawals per deposit, symlog)",
    ylabel="Hamming weight of deposit amount",
    title="Figure 4 (1-sum) - Deposit Hamming weight vs number of matches (color = density)",
    outpath=FIG_DIR + "knapsack1_fig4_scatter_deposit_hw_vs_matches.png",
    cmap="viridis",
)
 
# ================================================================
# FIGURE 5: density scatter  x = number of deposits explained by the withdraw, y = HW withdraw
# ================================================================
density_scatter(
    w_occ, w_hw,
    xlabel="Deposits explained per withdrawal (symlog)",
    ylabel="Hamming weight of withdrawal amount",
    title="Figure 5 (1-sum) - Withdrawal Hamming weight vs deposits explained (color = density)",
    outpath=FIG_DIR + "knapsack1_fig5_scatter_withdraw_hw_vs_deposits.png",
    cmap="viridis",
)
 
print("Figures saved to", FIG_DIR)
print(f"  deposits kept              : {len(dep_amounts)}")
print(f"  deposits with >=1 match    : {len(dep_amounts_with_match)}")
print(f"  distinct matching withdraws: {len(deposits_per_withdraw_vals)}")
print(f"  withdraws (aggregated)     : {n_w}")
 
if SHOW_PLOT:
    plt.show()