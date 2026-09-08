import csv
import bisect
from datetime import datetime, timezone
from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----
WITHDRAWS_CSV = "data/processed_privacypools_data_withdraws.csv"
# File produced by get_knapsack_3_iterations_dict.py
# (contains dep_time + linked_withdraws_count + truncated)
DEP_COUNTER_IN_PATH = "data/Heuristics/Knapsack/3-sum/knapsack_3_withdraws_triplets_per_deposit.csv"
S_SCORE_OUT_PATH = "data/Heuristics/Knapsack/3-sum/knapsack_3_withdraws_s_score.csv"
FIG_OUT_PATH = "figures/Heuristics/Knapsack/3-sum/knapsack3_s_score_hist.png"

SHOW_PLOT = False
SCALE = Decimal(1).scaleb(-9)  # 0.000000001


def q(val) -> int:
    return int((Decimal(str(val)) / SCALE).to_integral_value())


def parse_time(raw: str) -> datetime:
    s = raw.strip().replace(",", ".").replace(" UTC", "")
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)


# ================================================================
# 1. Load the withdraws (aggregate by tx_hash)
# ================================================================
raw_by_hash = {}
with open(WITHDRAWS_CSV, newline="") as f:
    for row in csv.DictReader(f):
        h = row["tx_hash"]
        amt = Decimal(str(row["eth_amount"]))
        if h in raw_by_hash:
            raw_by_hash[h]["amount"] += amt
        else:
            raw_by_hash[h] = {"amount": amt, "time": parse_time(row["evt_block_time"])}

withdraws = [{"amount_q": q(v["amount"]), "time": v["time"]} for v in raw_by_hash.values()]
print(f"Withdraws loaded (aggregated) : {len(withdraws)}")

# ================================================================
# 2. Pre-sort for the naive_anonymity_set
#    naive_anonymity_set(d) = number of withdraws such that
#         time_w > time_d   AND   amount_q_w < amount_q_d
#    (3-sum: each withdraw of a triplet is < d -> bisect_left, like 2-sum).
# ================================================================
withdraws.sort(key=lambda w: w["time"])
w_times = [w["time"] for w in withdraws]
w_amounts_q = [w["amount_q"] for w in withdraws]
n = len(withdraws)

suffix_sorted_amounts = [None] * (n + 1)
suffix_sorted_amounts[n] = []
running = []
for i in range(n - 1, -1, -1):
    bisect.insort(running, w_amounts_q[i])
    suffix_sorted_amounts[i] = list(running)

# ================================================================
# 3. Read the per-deposit file and compute s(d)
#    counter(d) = linked_withdraws_count (distinct linked withdraws).
#    Deposits TRUNCATED by the cutoff are excluded from the computation: their
#    linked_withdraws_count is a lower bound (the analysis stopped at the
#    threshold), so s(d) would be wrong. We write s_score="" for them.
# ================================================================
rows_out = []
n_truncated = 0
with open(DEP_COUNTER_IN_PATH, newline="") as f:
    for row in csv.DictReader(f):
        truncated = int(row["truncated"])
        dep_time = parse_time(row["dep_time"])
        dep_amount_q = q(float(row["dep_amount"]))
        linked = int(row["linked_withdraws_count"])

        start = bisect.bisect_right(w_times, dep_time)
        sorted_amts = suffix_sorted_amounts[start]
        naive_set = bisect.bisect_left(sorted_amts, dep_amount_q)

        if truncated:
            n_truncated += 1
            s_score = ""  # not computed for truncated deposits
        else:
            s_score = (linked / naive_set) if naive_set > 0 else 0.0

        rows_out.append({
            "dep_hash": row["dep_hash"],
            "dep_address": row["dep_address"],
            "dep_amount": row["dep_amount"],
            "dep_time": row["dep_time"],
            "linked_withdraws_count": linked,
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
        "linked_withdraws_count", "naive_anonymity_set", "truncated", "s_score",
    ])
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Deposits processed : {len(rows_out)}")
print(f"  excluded (truncated by cutoff) : {n_truncated}")
print(f"s(d) output saved  : {S_SCORE_OUT_PATH}")

# ================================================================
# 5. Histogram of s(d): non-truncated deposits with >=1 triplet
# ================================================================
s_scores = np.array([r["s_score"] for r in rows_out
                     if r["truncated"] == 0
                     and r["linked_withdraws_count"] > 0
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
plt.title("3-sum : distribution of s(d) over non-truncated deposits with >=1 triplet",
          fontsize=11)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_OUT_PATH, dpi=150)
print(f"Histogram saved    : {FIG_OUT_PATH}")

if SHOW_PLOT:
    plt.show()