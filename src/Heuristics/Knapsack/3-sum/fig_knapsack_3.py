import csv
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from matplotlib.colors import LogNorm

# ---- Paths ----
WITHDRAWS_CSV = "data/processed_privacypools_data_withdraws.csv"
# CSVs produced by get_knapsack_3_iterations_dict.py
DEP_COUNTER_PATH = "data/Heuristics/Knapsack/3-sum/knapsack_3_withdraws_triplets_per_deposit.csv"
TRIPLET_COUNTER_PATH = "data/Heuristics/Knapsack/3-sum/knapsack_3_withdraws_deposits_per_triplet.csv"
# detailed matches (dep_hash + the 3 w_hash): needed to cross discriminating
# triplets with the deposits they cover (figure 6)
MATCHES_PATH = "data/Heuristics/Knapsack/3-sum/knapsack_3_withdraws_matches.csv"
FIG_DIR = "figures/Heuristics/Knapsack/3-sum/"

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
dep_triplets_count = []
dep_amounts = []
dep_hw = []
dep_truncated = []
with open(DEP_COUNTER_PATH, newline="") as f:
    for row in csv.DictReader(f):
        dep_triplets_count.append(int(row["triplets_count"]))
        amt = float(row["dep_amount"])
        dep_amounts.append(amt)
        dep_hw.append(decimal_fraction_hamming_weight(amt))
        dep_truncated.append(int(row["truncated"]))

dep_triplets_count = np.array(dep_triplets_count)
dep_amounts = np.array(dep_amounts)
dep_hw = np.array(dep_hw)
dep_truncated = np.array(dep_truncated)

dep_amounts_with_triplet = dep_amounts[dep_triplets_count > 0]

# per-triplet + aggregation by withdraw (via tx_hash)
# occ_disc_per_withdraw : number of DISCRIMINATING triplets (explaining 1 deposit)
#                         in which each withdraw (tx_hash) appears
triplet_deposit_count = []
occ_disc_per_withdraw = {}
with open(TRIPLET_COUNTER_PATH, newline="") as f:
    for row in csv.reader(f):
        if row[0] == "w1_hash":
            continue
        h1, h2, h3, cnt = row[0], row[1], row[2], int(row[3])
        triplet_deposit_count.append(cnt)
        if cnt == 1:  # discriminating triplet
            for h in (h1, h2, h3):
                occ_disc_per_withdraw[h] = occ_disc_per_withdraw.get(h, 0) + 1

triplet_deposit_count = np.array(triplet_deposit_count)

# withdraws: amount + hw, indexed by tx_hash
withdraw_amounts = []
withdraw_hashes = []
with open(WITHDRAWS_CSV, newline="") as f:
    for row in csv.DictReader(f):
        withdraw_amounts.append(float(row["eth_amount"]))
        withdraw_hashes.append(row["tx_hash"])

withdraw_hw = [decimal_fraction_hamming_weight(a) for a in withdraw_amounts]
w_occ = np.array([occ_disc_per_withdraw.get(h, 0) for h in withdraw_hashes])
w_hw = np.array(withdraw_hw)


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
# FIGURE 1: histogram of the number of triplets per deposit (0 included)
#   NB: the "truncated" deposits (cutoff) are capped at MAX_LINKS; their bar
#   at the threshold is therefore a lower bound, not the true value.
# ================================================================
plt.figure(figsize=(10, 6))
max_tc = dep_triplets_count.max()
log_bins = np.logspace(0, np.log10(max_tc + 1), 60)
bins = np.concatenate(([0, 0.5], log_bins))
plt.hist(dep_triplets_count, bins=bins, color="#5b92e5",
         edgecolor="black", linewidth=0.5, linestyle=":")
plt.title("Figure 1 (3-sum) - Number of withdrawal triplets per deposit (including 0)",
          fontsize=11)
plt.xlabel("Number of matching 3-withdrawal triplets per deposit (symlog, capped by cutoff)")
plt.ylabel("Frequency (number of deposits)")
plt.yscale("log")
plt.xscale("symlog", linthresh=1)
plt.xlim(left=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR + "knapsack3_fig1_triplets_per_deposit_hist.png", dpi=150)

# ================================================================
# FIGURE 2: distribution of deposit amounts (all vs with triplet)
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
plt.hist(dep_amounts_with_triplet, bins=bins, color="#f07853", alpha=0.75,
         edgecolor="white", linewidth=0.3,
         label=f"Deposits with >=1 matching triplet (n={len(dep_amounts_with_triplet)})")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Deposit amount (ETH, net)")
plt.ylabel("# Deposits")
plt.title("Figure 2 (3-sum) - Deposit amount distribution: all kept vs. with matching triplet",
          fontsize=11)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR + "knapsack3_fig2_deposit_amounts_hist.png", dpi=150)

# ================================================================
# FIGURE 3: histogram of the per-triplet counter
# ================================================================
plt.figure(figsize=(10, 6))
max_tdc = triplet_deposit_count.max()
bins = np.logspace(0, np.log10(max_tdc + 1), 60)
plt.hist(triplet_deposit_count, bins=bins, color="#5b92e5",
         edgecolor="black", linewidth=0.5, linestyle=":")
plt.title("Figure 3 (3-sum) - Number of deposits explained per withdrawal triplet",
          fontsize=11)
plt.xlabel("Number of deposits matched by the same withdrawal triplet")
plt.ylabel("Frequency (number of triplets)")
plt.yscale("log"); plt.xscale("log")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR + "knapsack3_fig3_deposits_per_triplet_hist.png", dpi=150)

# ================================================================
# FIGURE 4: density scatter  x = triplet counter (deposit), y = HW deposit
# ================================================================
density_scatter(
    dep_triplets_count, dep_hw,
    xlabel="3-sum triplet counter (triplets per deposit, symlog, capped)",
    ylabel="Hamming weight of deposit amount",
    title="Figure 4 (3-sum) - Deposit Hamming weight vs number of 3-sum triplets (color = density)",
    outpath=FIG_DIR + "knapsack3_fig4_scatter_deposit_hw_vs_triplets.png",
    cmap="viridis",
)

# ================================================================
# FIGURE 5: density scatter  x = occurrences in discriminating triplets, y = HW withdraw
# ================================================================
density_scatter(
    w_occ, w_hw,
    xlabel="Occurrence counter in discriminating triplets (x=1 triplets only, symlog)",
    ylabel="Hamming weight of withdrawal amount",
    title="Figure 5 (3-sum) - Withdrawal Hamming weight vs occurrences in discriminating triplets (color = density)",
    outpath=FIG_DIR + "knapsack3_fig5_scatter_withdraw_hw_vs_occurrences.png",
    cmap="viridis",
)

# ================================================================
# FIGURE 6: concentration of discriminating triplets on deposits
#
#   A DISCRIMINATING triplet explains a single deposit (deposits_count == 1).
#   This figure shows that, despite their high number, these triplets cover
#   only a small number of distinct deposits: a single deposit is the target
#   of hundreds of competing discriminating triplets. "Discrimination" at the
#   triplet level therefore does NOT translate into identification at the
#   deposit level.
#
#   (a) contrast bars: number of discriminating triplets vs number of deposits covered
#   (b) distribution of the number of discriminating triplets per deposit
# ================================================================
from collections import Counter as _Counter

# discriminating triplets (key = 3 sorted tx_hash)
disc_triplets = set()
with open(TRIPLET_COUNTER_PATH, newline="") as f:
    for row in csv.reader(f):
        if row[0] == "w1_hash":
            continue
        if int(row[3]) == 1:
            disc_triplets.add(tuple(sorted((row[0], row[1], row[2]))))

# how many discriminating triplets cover each deposit
disc_per_deposit = _Counter()
with open(MATCHES_PATH, newline="") as f:
    for row in csv.DictReader(f):
        key = tuple(sorted((row["w1_hash"], row["w2_hash"], row["w3_hash"])))
        if key in disc_triplets:
            disc_per_deposit[row["dep_hash"]] += 1

disc_vals = np.array(list(disc_per_deposit.values()))
n_disc_triplets = len(disc_triplets)
n_deps_covered = len(disc_vals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- (a) contrast triplets vs deposits ---
ax1.bar(["Discriminating\ntriplets", "Distinct deposits\ncovered"],
        [n_disc_triplets, n_deps_covered],
        color=["#5b92e5", "#f07853"], edgecolor="black", width=0.6)
ax1.set_yscale("log")
ax1.set_ylabel("Count (log scale)")
ax1.set_ylim(top=n_disc_triplets * 3)   # margin above the bar for the label
ax1.set_title(f"(a) {n_disc_triplets:,} discriminating triplets cover only {n_deps_covered} deposits",
              fontsize=11)
for i, v in enumerate([n_disc_triplets, n_deps_covered]):
    ax1.text(i, v * 1.15, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# --- (b) distribution per deposit ---
ax2.hist(disc_vals, bins=range(1, disc_vals.max() + 2),
         color="#f07853", edgecolor="black", linewidth=0.4)
ax2.set_yscale("log")
ax2.set_xlabel("Number of discriminating triplets per deposit")
ax2.set_ylabel("Frequency (number of deposits)")
ax2.set_title(f"(b) Discriminating triplets per deposit (cut-off = {int(disc_vals.max())})",
              fontsize=11)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Figure 6 (3-sum) - Discriminating triplets concentrate on few deposits",
             fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR + "knapsack3_fig6_discriminating_triplets_concentration.png", dpi=150)

print("Figures saved to", FIG_DIR)
print(f"  deposits kept              : {len(dep_amounts)}")
print(f"  deposits with >=1 triplet  : {len(dep_amounts_with_triplet)}")
print(f"  deposits truncated (cutoff): {int(dep_truncated.sum())}")
print(f"  distinct matching triplets : {len(triplet_deposit_count)}")
print(f"  discriminating triplets    : {n_disc_triplets}")
print(f"  deposits covered by them   : {n_deps_covered}")
print(f"  withdraws                  : {len(withdraw_amounts)}")

if SHOW_PLOT:
    plt.show()