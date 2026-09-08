import csv
import bisect
from datetime import datetime, timezone
from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----
WITHDRAWS_CSV = "data/processed_privacypools_data_withdraws.csv"
# File produced by knapsack_2_withdraws_dict.py (contains dep_time + pairs_count)
DEP_COUNTER_IN_PATH = "data/Heuristics/Knapsack/knapsack_2_withdraws_pairs_per_deposit.csv"
# Enriched output: adds naive_anonymity_set and s_score
S_SCORE_OUT_PATH = "data/Heuristics/Knapsack/2-sum/knapsack_2_withdraws_s_score.csv"
# Figure: histogram of s(d)
FIG_OUT_PATH = "figures/Heuristics/Knapsack/2-sum/knapsack_s_score_hist.png"
# Figure: s(d) as a function of deposit time
FIG_TIME_OUT_PATH = "figures/Heuristics/Knapsack/2-sum/knapsack_s_score_vs_time.png"

 
SHOW_PLOT = False  # True to display on screen in addition to saving
 
SCALE = Decimal(1).scaleb(-9)  # 0.000000001
 
 
def q(val) -> int:
    """ETH amount -> integer key (units of 1e-9)."""
    return int((Decimal(str(val)) / SCALE).to_integral_value())
 
 
def parse_time(raw: str) -> datetime:
    """Parse a dataset timestamp. Handles the decimal comma and the ' UTC' suffix."""
    s = raw.strip().replace(",", ".").replace(" UTC", "")
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
 
 
# ================================================================
# 1. Load the withdraws (quantized amount + timestamp)
# ================================================================
withdraws = []
with open(WITHDRAWS_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        amount = float(row["eth_amount"])
        withdraws.append({
            "amount_q": q(amount),
            "time": parse_time(row["evt_block_time"]),
        })
 
print(f"Withdraws loaded : {len(withdraws)}")
 
# ================================================================
# 2. Pre-sort for a fast naive_anonymity_set computation
#
#    naive_anonymity_set(d) = number of withdraws such that
#         time_w > time_d   AND   amount_q_w < amount_q_d
#
#    Method: sort the withdraws by timestamp. For a given deposit, all the
#    "later" withdraws form a contiguous suffix of this sorted list (found
#    in O(log n) by bisecting on the times). Within that suffix, it remains
#    to count those whose amount < deposit.
#
#    To avoid re-scanning the suffix for each deposit, we precompute, for
#    each time-cut position, the SORTED list of the suffix amounts
#    (snapshot). The final count is then done with bisect.
# ================================================================
withdraws.sort(key=lambda w: w["time"])
w_times = [w["time"] for w in withdraws]
w_amounts_q = [w["amount_q"] for w in withdraws]
n = len(withdraws)
 
# suffix_sorted_amounts[i] = sorted list of amounts of withdraws with index >= i
suffix_sorted_amounts = [None] * (n + 1)
suffix_sorted_amounts[n] = []
running = []
for i in range(n - 1, -1, -1):
    bisect.insort(running, w_amounts_q[i])
    suffix_sorted_amounts[i] = list(running)  # snapshot
 
# ================================================================
# 3. Read the per-deposit file and compute s(d)
# ================================================================
rows_out = []
with open(DEP_COUNTER_IN_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        dep_time = parse_time(row["dep_time"])
        dep_amount_q = q(float(row["dep_amount"]))
        pairs_count = int(row["pairs_count"])
        # counter(d) = number of DISTINCT withdraws involved in >=1 pair
        # (computed by the knapsack script). This is the correct numerator for
        # s(d): these withdraws are a subset of the naive set, so s(d) <= 1.
        linked_withdraws = int(row["linked_withdraws_count"])
        # Deposits truncated by the cutoff: linked_withdraws_count is a lower
        # bound (the analysis stopped at the threshold), so s(d) would be wrong -> excluded.
        truncated = int(row.get("truncated", 0))
 
        # Index of the first withdraw STRICTLY later than the deposit
        start = bisect.bisect_right(w_times, dep_time)
 
        # In the suffix [start:], count the amounts < dep_amount_q
        sorted_amts = suffix_sorted_amounts[start]
        naive_set = bisect.bisect_left(sorted_amts, dep_amount_q)
 
        if truncated:
            s_score = ""   # not computed for truncated deposits
        else:
            s_score = (linked_withdraws / naive_set) if naive_set > 0 else 0.0
 
        rows_out.append({
            "dep_hash": row["dep_hash"],
            "dep_address": row["dep_address"],
            "dep_amount": row["dep_amount"],
            "dep_time": row["dep_time"],
            "pairs_count": pairs_count,
            "linked_withdraws_count": linked_withdraws,
            "naive_anonymity_set": naive_set,
            "truncated": truncated,
            "s_score": s_score,
        })
 
# ================================================================
# 4. Write the CSV
# ================================================================
with open(S_SCORE_OUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dep_hash", "dep_address", "dep_amount", "dep_time",
        "pairs_count", "linked_withdraws_count",
        "naive_anonymity_set", "truncated", "s_score",
    ])
    writer.writeheader()
    writer.writerows(rows_out)
 
print(f"Deposits processed : {len(rows_out)}")
print(f"s(d) output saved  : {S_SCORE_OUT_PATH}")
 
# ================================================================
# 5. Histogram of s(d)
#
#    We keep only the deposits with at least one pair (pairs_count > 0) AND
#    NOT truncated by the cutoff: a truncated deposit has an underestimated
#    linked_withdraws_count, so its s(d) would be wrong and understated.
#
#    s(d) = (number of distinct linked withdraws) / (naive anonymity set) is in
#    [0, 1] by construction: the linked withdraws are a subset of the naive
#    set. The distribution spans several orders of magnitude, so LOGARITHMIC
#    bins + log x-axis. The line at s=1 marks the boundary case where ALL the
#    deposit's plausible withdrawals are involved in at least one pair (the
#    least protected deposit).
# ================================================================
s_scores = np.array([r["s_score"] for r in rows_out
                     if r["truncated"] == 0
                     and r["pairs_count"] > 0
                     and r["s_score"] not in ("", 0.0)])
 
print(f"Deposits kept for histogram : {len(s_scores)}")
if len(s_scores):
    print(f"  s(d) min : {s_scores.min():.6g}   max : {s_scores.max():.6g}   median : {np.median(s_scores):.6g}")
 
plt.figure(figsize=(10, 6))
bins = np.logspace(np.log10(s_scores.min()), np.log10(s_scores.max()), 60)
plt.hist(s_scores, bins=bins, color="#5b92e5", edgecolor="black", linewidth=0.4)
plt.axvline(1.0, color="#f07853", linestyle="--", linewidth=1.5,
            label="s(d) = 1 (all plausible withdrawals are linked)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("s(d) = linked_withdraws_count(d) / naive_anonymity_set(d)")
plt.ylabel("Frequency (number of deposits)")
plt.title("Distribution of s(d) over deposits with at least one matching pair",
          fontsize=11)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_OUT_PATH, dpi=150)
print(f"Histogram saved    : {FIG_OUT_PATH}")
 
# ================================================================
# 6. s(d) as a function of deposit time
#
#    Two panels sharing the time axis:
#      (top)    s(d): shows whether the score drifts over time.
#      (bottom) counter(d) and naive_anonymity_set(d) overlaid: illustrates
#               that both quantities decrease together as we move forward in
#               the dataset (fewer later withdraws remaining), with counter
#               staying bounded by the naive set.
#    Only deposits with at least one pair are plotted.
# ================================================================
ts = []
sc = []
cnt = []
nav = []
for r in rows_out:
    if r["pairs_count"] > 0:
        ts.append(parse_time(r["dep_time"]))
        sc.append(r["s_score"])
        cnt.append(r["linked_withdraws_count"])
        nav.append(r["naive_anonymity_set"])
 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
 
# --- top panel: s(d) ---
ax1.scatter(ts, sc, s=10, alpha=0.4, c="#5b92e5", edgecolors="none")
ax1.axhline(1.0, color="#f07853", linestyle="--", linewidth=1.2,
            label="s(d) = 1")
ax1.set_ylabel("s(d)")
ax1.set_title("s(d) and its components over deposit time (deposits with >=1 pair)",
              fontsize=11)
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.4)
 
# --- bottom panel: counter(d) and naive set ---
ax2.scatter(ts, nav, s=10, alpha=0.4, c="#9aa7b8", edgecolors="none",
            label="naive anonymity set(d)")
ax2.scatter(ts, cnt, s=10, alpha=0.5, c="#f07853", edgecolors="none",
            label="linked withdraws count(d) = counter(d)")
ax2.set_yscale("log")
ax2.set_ylabel("Count (log scale)")
ax2.set_xlabel("Deposit time")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.4)
 
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_TIME_OUT_PATH, dpi=150)
print(f"s(d) vs time saved : {FIG_TIME_OUT_PATH}")
 
if SHOW_PLOT:
    plt.show()