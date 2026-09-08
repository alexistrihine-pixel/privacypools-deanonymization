import csv
from datetime import datetime, timezone
from decimal import Decimal

# ---- Paths to input data (edit if your files live elsewhere) ----
DEPOSITS_CSV = "data/processed_privacypools_eth_pool_deposits.csv"
WITHDRAWS_CSV = "data/processed_privacypools_data_withdraws.csv"
NO_BALANCE_DEPOSITORS_PATH = "data/Heuristics/Address_Reuse/no_balance_depositors.txt"
ONE_SUM_COUNTER_PATH = "data/Heuristics/Knapsack/1-sum/knapsack_1_withdraw_matches_per_deposit.csv"
MATCHES_OUT_PATH = "data/Heuristics/Knapsack/3-sum/knapsack_3_withdraws_matches.csv"
# Counter outputs (for the histograms)
DEP_COUNTER_OUT_PATH = "data/Heuristics/Knapsack/3-sum/knapsack_3_withdraws_triplets_per_deposit.csv"
DEP_COUNTER_ALL_OUT_PATH = "data/Heuristics/Knapsack/3-sum/knapsack_3_withdraws_triplets_per_deposit_all.csv"
TRIPLET_COUNTER_OUT_PATH = "data/Heuristics/Knapsack/3-sum/knapsack_3_withdraws_deposits_per_triplet.csv"

# ---- Config ----
TOLERANCE = 1e-9          # tight tolerance: above it we would group distinct amounts
MIN_DEPOSIT_HW = 2        # keep only deposits with a "complex" amount
# Cutoff: as soon as a deposit reaches this number of triplets, we drop it and
# move on to the next. Avoids the combinatorial explosion of the 3-sum on
# "easy" deposits. The criterion is the NUMBER OF LINKS FOUND (not the naive
# set, which would bias toward old deposits having many later withdraws).
MAX_LINKS_PER_DEPOSIT = 200
 
SCALE = Decimal(1).scaleb(-9)  # 0.000000001
 
 
def q(val) -> int:
    """ETH amount -> integer key (units of 1e-9)."""
    return int((Decimal(str(val)) / SCALE).to_integral_value())
 
 
def parse_time(raw: str) -> datetime:
    """Parse a dataset timestamp. Handles the decimal comma and the ' UTC' suffix."""
    s = raw.strip().replace(",", ".").replace(" UTC", "")
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
 
 
def decimal_fraction_hamming_weight(val) -> int:
    """Number of non-zero fractional digits (base-10 Hamming weight)."""
    d = Decimal(str(val)).normalize()
    parts = f"{d:f}".split(".")
    if len(parts) < 2:
        return 0
    return sum(1 for ch in parts[1].rstrip("0") if ch != "0")
 
 
# ================================================================
# 1. Load the already-leaked depositors (to exclude)
# ================================================================
with open(NO_BALANCE_DEPOSITORS_PATH, "r") as f:
    no_balance_depositors = set(line.strip().lower() for line in f if line.strip())
 
# 1b. Deposits already resolved by the 1-sum (unique exact match): we exclude them.
one_sum_resolved = set()
with open(ONE_SUM_COUNTER_PATH, newline="") as f:
    for row in csv.DictReader(f):
        if int(row["matches_count"]) == 1:
            one_sum_resolved.add(row["dep_hash"])
print(f"Deposits resolved by 1-sum (matches_count==1): {len(one_sum_resolved)}")
 
# ================================================================
# 2. Load the deposits
# ================================================================
deposits = []
n_total_dep = 0
with open(DEPOSITS_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        n_total_dep += 1
        depositor = row["depositor"].strip().lower()
        # We KEEP the 1-sum filter (deposits already resolved by unique exact match).
        if row["tx_hash"] in one_sum_resolved:
            continue
        amount_net = float(row["amount_eth_net"])
        if MIN_DEPOSIT_HW > 0 and decimal_fraction_hamming_weight(amount_net) < MIN_DEPOSIT_HW:
            continue
        # Address-reuse deposits loaded too (marked "excluded") to produce
        # a 2nd complete per_deposit file.
        deposits.append({
            "dep_id": len(deposits),
            "depositor": depositor,
            "tx_hash": row["tx_hash"],
            "amount_net": amount_net,
            "amount_q": q(amount_net),
            "time": parse_time(row["time"]),
            "excluded": depositor in no_balance_depositors,
        })
 
n_excluded = sum(1 for d in deposits if d["excluded"])
 
print(f"Deposits total in file        : {n_total_dep}")
print(f"Deposits loaded (HW>={MIN_DEPOSIT_HW}, 1-sum filtered, address-reuse included): {len(deposits)}")
print(f"  of which kept (address-reuse excluded): {len(deposits) - n_excluded}")
 
# ================================================================
# 3. Load the withdraws (list indexed by w_id)
# ================================================================
withdraws = []
# Prior aggregation by tx_hash: several rows may share the same tx_hash
# (one on-chain transaction emitting several outputs to the SAME address).
# We merge them into a single withdraw whose amount is the SUM.
raw_by_hash = {}
with open(WITHDRAWS_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        h = row["tx_hash"]
        amt = Decimal(str(row["eth_amount"]))
        if h in raw_by_hash:
            raw_by_hash[h]["amount"] += amt
        else:
            raw_by_hash[h] = {
                "amount": amt,
                "time": parse_time(row["evt_block_time"]),
                "recipient": row["recipient"].strip().lower(),
                "tx_hash": h,
            }
 
for h, agg in raw_by_hash.items():
    amount = float(agg["amount"])
    withdraws.append({
        "w_id": len(withdraws),
        "amount": amount,
        "amount_q": q(agg["amount"]),
        "time": agg["time"],
        "recipient": agg["recipient"],
        "tx_hash": h,
    })
 
n_w = len(withdraws)
print(f"Withdraws loaded (aggregated) : {n_w}")
 
# Fast index w_id -> withdraw (to reconstruct a pair's info)
w_by_id = {w["w_id"]: w for w in withdraws}
 
# ================================================================
# 4. Pre-fill the 2Sum dictionary
#
#    two_sum[sum_q] = list of pairs, each pair stored compactly:
#        (w_id_i, w_id_j, min_time)
#    - sum_q : sum of the two amounts (integer key in 1e-9)
#    - min_time : the EARLIEST timestamp of the two withdraws. A pair is valid
#      for a deposit d if min_time > d.time (BOTH withdraws must be later than
#      d, so the earliest one is enough to check).
#
#    We build ALL pairs (i < j): ~n_w^2/2 entries.
# ================================================================
two_sum = {}
for i in range(n_w):
    wi = withdraws[i]
    ai = wi["amount_q"]; ti = wi["time"]; idi = wi["w_id"]
    for j in range(i + 1, n_w):
        wj = withdraws[j]
        s = ai + wj["amount_q"]
        min_t = ti if ti < wj["time"] else wj["time"]
        two_sum.setdefault(s, []).append((idi, wj["w_id"], min_t))
 
print(f"2Sum dictionary keys         : {len(two_sum)}")
 
# ================================================================
# 5. Triplet search: d == Wz + Wi + Wj
#
#    For each deposit d, for each withdraw Wz that is later and has amount
#    < d: we look up the complementary sum (d - Wz) in two_sum. Each pair
#    (Wi, Wj) found, if temporally valid and with tx_hash distinct from Wz,
#    forms a triplet.
#
#    Deduplication: a triplet {A, B, C} can be found up to 3 times (once per
#    element playing the role of Wz). We keep a sorted key of the 3 w_id in a
#    per-deposit set so as to count each triplet only once.
#
#    Cutoff: as soon as the number of distinct triplets of the deposit exceeds
#    MAX_LINKS_PER_DEPOSIT, we drop this deposit ("easy" deposit).
# ================================================================
matches = []
triplets_per_deposit = {dep["dep_id"]: 0 for dep in deposits}
deposit_truncated = {dep["dep_id"]: False for dep in deposits}
distinct_withdraws_per_deposit = {dep["dep_id"]: set() for dep in deposits}
deposits_per_triplet = {}   # key = sorted (w1_id,w2_id,w3_id) -> number of deposits
 
 
def record_triplet(dep, wz, wi, wj):
    diff_dec = abs(Decimal(str(wz["amount"])) + Decimal(str(wi["amount"]))
                   + Decimal(str(wj["amount"])) - Decimal(str(dep["amount_net"])))
    matches.append({
        "dep_hash": dep["tx_hash"], "dep_address": dep["depositor"], "dep_amount": dep["amount_net"],
        "w1_hash": wz["tx_hash"], "w1_address": wz["recipient"], "w1_amount": wz["amount"],
        "w2_hash": wi["tx_hash"], "w2_address": wi["recipient"], "w2_amount": wi["amount"],
        "w3_hash": wj["tx_hash"], "w3_address": wj["recipient"], "w3_amount": wj["amount"],
        "difference": float(diff_dec),
        "_excluded": dep["excluded"],   # interne : filtre a l'ecriture
    })
    triplets_per_deposit[dep["dep_id"]] += 1
    distinct_withdraws_per_deposit[dep["dep_id"]].update(
        (wz["tx_hash"], wi["tx_hash"], wj["tx_hash"]))
    # per-triplet counter: FILTERED population (for the figures)
    if not dep["excluded"]:
        key = tuple(sorted((wz["tx_hash"], wi["tx_hash"], wj["tx_hash"])))
        deposits_per_triplet[key] = deposits_per_triplet.get(key, 0) + 1
 
 
for dep in deposits:
    dep_q = dep["amount_q"]
    dep_time = dep["time"]
    dep_id = dep["dep_id"]
    seen_triplets = set()   # keys of triplets already counted for this deposit
    stop = False
 
    for wz in withdraws:
        if wz["time"] <= dep_time:
            continue
        if wz["amount_q"] >= dep_q:   # Wz alone cannot exceed d
            continue
 
        need = dep_q - wz["amount_q"]   # target sum for the pair (Wi, Wj)
        pairs = two_sum.get(need)
        if not pairs:
            continue
 
        for (idi, idj, min_t) in pairs:
            # pair posteriority: the earlier of the two > d
            if min_t <= dep_time:
                continue
            wi = w_by_id[idi]
            wj = w_by_id[idj]
            # 3 distinct tx_hash (one on-chain transaction can have several
            # w_id; we distinguish and deduplicate by tx_hash)
            hz, hi, hj = wz["tx_hash"], wi["tx_hash"], wj["tx_hash"]
            if hz == hi or hz == hj or hi == hj:
                continue
 
            key = tuple(sorted((hz, hi, hj)))
            if key in seen_triplets:
                continue   # already counted (via another Wz, or same tx_hash)
            seen_triplets.add(key)
 
            record_triplet(dep, wz, wi, wj)
 
            if triplets_per_deposit[dep_id] >= MAX_LINKS_PER_DEPOSIT:
                deposit_truncated[dep_id] = True
                stop = True
                break
        if stop:
            break
 
# Carry over to each deposit
for dep in deposits:
    dep["triplets_count"] = triplets_per_deposit[dep["dep_id"]]
    dep["linked_withdraws_count"] = len(distinct_withdraws_per_deposit[dep["dep_id"]])
    dep["truncated"] = deposit_truncated[dep["dep_id"]]
 
# ================================================================
# 6. Terminal results
# ================================================================
print(f"\nTotal triplets recorded       : {len(matches)}")
n_with_triplet = sum(1 for d in deposits if d['triplets_count'] > 0)
n_no_triplet = sum(1 for d in deposits if d['triplets_count'] == 0)
n_dep = len(deposits)
pct_with_triplet = (100 * n_with_triplet / n_dep) if n_dep else 0.0
pct_no_triplet = (100 * n_no_triplet / n_dep) if n_dep else 0.0
print(f"Deposits with >=1 triplet     : {n_with_triplet}  ({pct_with_triplet:.1f}% of {n_dep} loaded)")
print(f"Deposits with 0 triplet       : {n_no_triplet}  ({pct_no_triplet:.1f}% of {n_dep} loaded)")
print(f"Deposits truncated (cutoff)   : {sum(1 for d in deposits if d['truncated'])}")
print(f"Distinct matching triplets    : {len(deposits_per_triplet)}")
 
# ================================================================
# 7. Write the CSVs
# ================================================================
# 7a. Detailed triplets
fieldnames = [
    "dep_hash", "dep_address", "dep_amount",
    "w1_hash", "w1_address", "w1_amount",
    "w2_hash", "w2_address", "w2_amount",
    "w3_hash", "w3_address", "w3_amount",
    "difference",
]
with open(MATCHES_OUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(m for m in matches if not m["_excluded"])
print(f"\nMatches saved to           {MATCHES_OUT_PATH}")
 
# 7b. Triplets-per-deposit counter (zeros included)
#     Two files, same format:
#       - DEP_COUNTER_OUT_PATH     : FILTERED (address-reuse excluded), figures + s(d).
#       - DEP_COUNTER_ALL_OUT_PATH : address-reuse INCLUDED, total anonymity loss.
dep_fields = [
    "dep_hash", "dep_address", "dep_amount", "dep_time",
    "triplets_count", "linked_withdraws_count", "truncated",
]
 
 
def write_per_deposit(path, include_excluded):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dep_fields)
        writer.writeheader()
        for dep in deposits:
            if dep["excluded"] and not include_excluded:
                continue
            writer.writerow({
                "dep_hash": dep["tx_hash"],
                "dep_address": dep["depositor"],
                "dep_amount": dep["amount_net"],
                "dep_time": dep["time"].strftime("%Y-%m-%d %H:%M:%S.%f") + " UTC",
                "triplets_count": dep["triplets_count"],
                "linked_withdraws_count": dep["linked_withdraws_count"],
                "truncated": int(dep["truncated"]),
            })
 
 
write_per_deposit(DEP_COUNTER_OUT_PATH, include_excluded=False)
print(f"Per-deposit counter (filtered) saved  {DEP_COUNTER_OUT_PATH}")
write_per_deposit(DEP_COUNTER_ALL_OUT_PATH, include_excluded=True)
print(f"Per-deposit counter (ALL)      saved  {DEP_COUNTER_ALL_OUT_PATH}")
 
# 7c. Deposits-per-withdraw-triplet counter (key = 3 tx_hash)
with open(TRIPLET_COUNTER_OUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["w1_hash", "w2_hash", "w3_hash", "deposits_count"])
    for (h1, h2, h3), cnt in deposits_per_triplet.items():
        writer.writerow([h1, h2, h3, cnt])
print(f"Per-triplet counter saved  {TRIPLET_COUNTER_OUT_PATH}")