import csv
from datetime import datetime, timezone
from decimal import Decimal

# ---- Paths to input data (edit if your files live elsewhere) ----
DEPOSITS_CSV = "data/processed_privacypools_eth_pool_deposits.csv"
WITHDRAWS_CSV = "data/processed_privacypools_data_withdraws.csv"
NO_BALANCE_DEPOSITORS_PATH = "data/Heuristics/Address_Reuse/no_balance_depositors.txt"
ONE_SUM_COUNTER_PATH = "data/Heuristics/Knapsack/1-sum/knapsack_1_withdraw_matches_per_deposit.csv"
MATCHES_OUT_PATH = "data/Heuristics/Knapsack/2-sum/knapsack_2_withdraws_matches.csv"
# Counter outputs (for the histograms)
DEP_COUNTER_OUT_PATH = "data/Heuristics/Knapsack/2-sum/knapsack_2_withdraws_pairs_per_deposit.csv"
DEP_COUNTER_ALL_OUT_PATH = "data/Heuristics/Knapsack/2-sum/knapsack_2_withdraws_pairs_per_deposit_all.csv"
PAIR_COUNTER_OUT_PATH = "data/Heuristics/Knapsack/2-sum/knapsack_2_withdraws_deposits_per_pair.csv"


# ---- Config ----
TOLERANCE = 1e-9      # tight tolerance: above it we would group distinct amounts (false positives)
MIN_DEPOSIT_HW = 2    # keep only deposits with a "complex" amount (>= this number of non-zero digits)
# Cutoff: as soon as a deposit reaches this number of PAIRS, we drop it and
# move on to the next (as for the 3-sum). Avoids collecting huge numbers of
# pairs on "easy" deposits (round amounts) which are not identifiable anyway.
# The criterion is the NUMBER OF LINKS FOUND. Set to None to disable the
# cutoff (original behavior, no limit).
MAX_LINKS_PER_DEPOSIT = 200

# scale: we convert each ETH amount into an INTEGER in units of 1e-9.
# Two amounts differing by less than TOLERANCE fall on the same integer
# => we compare integer keys, never suffering from float imprecision.
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
#     A deposit with matches_count == 1 in 1-sum already has a strong established
#     link; no need to analyze it in 2-sum. We identify it by dep_hash (stable id).
one_sum_resolved = set()
with open(ONE_SUM_COUNTER_PATH, newline="") as f:
    for row in csv.DictReader(f):
        if int(row["matches_count"]) == 1:
            one_sum_resolved.add(row["dep_hash"])
print(f"Deposits resolved by 1-sum (matches_count==1): {len(one_sum_resolved)}")

# ================================================================
# 2. Load the deposits (list of dicts)
#    We exclude the leaked depositors and filter by Hamming weight.
#    Each deposit gets a unique id (dep_id) = its index in the list, used as
#    the counter key so that we can KEEP the zeros.
# ================================================================
deposits = []
n_total_dep = 0
with open(DEPOSITS_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        n_total_dep += 1
        depositor = row["depositor"].strip().lower()
        # On GARDE le filtre 1-sum (deposits deja resolus par match exact unique).
        if row["tx_hash"] in one_sum_resolved:
            continue
        amount_net = float(row["amount_eth_net"])
        if MIN_DEPOSIT_HW > 0 and decimal_fraction_hamming_weight(amount_net) < MIN_DEPOSIT_HW:
            continue
        # We load the address-reuse deposits too, marked "excluded", to
        # produce a 2nd complete per_deposit file. The flag lets us write
        # one filtered (without address-reuse), the other with.
        deposits.append({
            "dep_id": len(deposits),             # unique id = index in the list
            "depositor": depositor,
            "tx_hash": row["tx_hash"],
            "amount_net": amount_net,
            "amount_q": q(amount_net),           # integer key of the amount
            "time": parse_time(row["time"]),
            "excluded": depositor in no_balance_depositors,
        })

n_excluded = sum(1 for d in deposits if d["excluded"])

print(f"Deposits total in file        : {n_total_dep}")
print(f"Deposits loaded (HW>={MIN_DEPOSIT_HW}, 1-sum filtered, address-reuse included): {len(deposits)}")
print(f"  of which kept (address-reuse excluded): {len(deposits) - n_excluded}")

# ================================================================
# 3. Load the withdraws and INDEX by amount
#
#    withdraws_by_amount : key = rounded_amount (integer 1e-9)
#                          value = list of withdraws with that amount
#
#    This is the heart of the optimization: instead of re-scanning all the
#    withdraws for each deposit, we bucket them ONCE by amount. Each withdraw
#    gets a unique id (w_id) to identify a pair in a stable way (used as the
#    key of the "deposits per pair" counter).
# ================================================================
withdraws_by_amount = {}
n_withdraws = 0
# Prior aggregation by tx_hash: several rows may share the same tx_hash
# (one on-chain transaction emitting several outputs to the SAME address).
# We merge them into a single withdraw whose amount is the SUM (verified:
# same recipient / relayer / timestamp within each group).
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
    key = q(agg["amount"])
    tx = {
        "w_id": n_withdraws,                 # unique id of the aggregated withdraw
        "amount": amount,
        "time": agg["time"],
        "recipient": agg["recipient"],
        "tx_hash": h,
    }
    withdraws_by_amount.setdefault(key, []).append(tx)
    n_withdraws += 1

print(f"Withdraws indexed (aggregated): {n_withdraws}")
print(f"Distinct withdraw amounts     : {len(withdraws_by_amount)}")

# SORTED list of amount keys: enables the two-pointer sweep.
sorted_keys = sorted(withdraws_by_amount.keys())

# ================================================================
# 3b. Initialize the counters (with the zeros)
#
#   pairs_per_deposit : dep_id -> number of withdraw pairs matching this deposit.
#       Initialized to 0 for ALL kept deposits (the zeros are preserved because
#       a deposit with no pair at all is information for the histogram).
#
#   deposits_per_pair : (w1_id, w2_id) -> number of distinct deposits that this
#       same withdraw pair can explain.
# ================================================================
pairs_per_deposit = {dep["dep_id"]: 0 for dep in deposits}
deposits_per_pair = {}
# distinct_withdraws_per_deposit : dep_id -> set of distinct w_id involved in
# at least one pair summing to this deposit. Its size is the "counter(d)" used
# for s(d) = counter(d) / naive_anonymity_set(d). These withdraws are a subset
# of the naive set (all < deposit and later), so counter(d) <= naive_set is
# guaranteed => s(d) in [0, 1].
distinct_withdraws_per_deposit = {dep["dep_id"]: set() for dep in deposits}
# deposit_truncated : dep_id -> True if the deposit reached the cutoff
deposit_truncated = {dep["dep_id"]: False for dep in deposits}

# ================================================================
# 4. Pair search: deposit == withdraw_1 + withdraw_2
#
#    Two-pointer over sorted_keys. Once a pair of AMOUNTS is found, we unpack
#    the lists of real transactions and apply the temporal filter (withdraw
#    AFTER the deposit).
#
#    The "difference" is computed in Decimal (exact decimal arithmetic) to
#    avoid float noise of the 2.7e-17 kind on sums that equal the deposit
#    exactly.
# ================================================================
matches = []
tol_q = q(TOLERANCE)  # tolerance expressed in key units (here ~1)


def record_match(dep, w1, w2):
    """Record a pair (w1, w2) for deposit dep + update the counters."""
    # exact difference in Decimal
    diff_dec = abs(Decimal(str(w1["amount"])) + Decimal(str(w2["amount"]))
                   - Decimal(str(dep["amount_net"])))
    matches.append({
        "dep_hash": dep["tx_hash"], "dep_address": dep["depositor"], "dep_amount": dep["amount_net"],
        "w1_hash": w1["tx_hash"], "w1_address": w1["recipient"], "w1_amount": w1["amount"],
        "w2_hash": w2["tx_hash"], "w2_address": w2["recipient"], "w2_amount": w2["amount"],
        "difference": float(diff_dec),
        "_excluded": dep["excluded"],   # interne : filtre a l'ecriture
    })
    # Counter 1: number of pairs per deposit
    pairs_per_deposit[dep["dep_id"]] += 1
    # Counter 3: DISTINCT withdraws involved in a pair for this deposit
    distinct_withdraws_per_deposit[dep["dep_id"]].add(w1["w_id"])
    distinct_withdraws_per_deposit[dep["dep_id"]].add(w2["w_id"])
    # Counter 2: number of deposits per pair (FILTERED population, for the figures)
    if not dep["excluded"]:
        pair_key = tuple(sorted((w1["w_id"], w2["w_id"])))
        deposits_per_pair[pair_key] = deposits_per_pair.get(pair_key, 0) + 1


for dep in deposits:
    dep_q = dep["amount_q"]
    dep_time = dep["time"]
    dep_id = dep["dep_id"]
    stop = False   # becomes True when the cutoff is reached

    lo, hi = 0, len(sorted_keys) - 1
    while lo <= hi:
        k_lo = sorted_keys[lo]
        k_hi = sorted_keys[hi]

        # We ignore amounts >= the deposit (a single withdraw cannot exceed
        # the deposit in a sum of 2 positive terms).
        if k_hi >= dep_q:
            hi -= 1
            continue

        s = k_lo + k_hi
        diff = s - dep_q

        if abs(diff) <= tol_q:
            list_lo = withdraws_by_amount[k_lo]
            list_hi = withdraws_by_amount[k_hi]

            if lo == hi:
                # Case a1 == a2: pairs WITHIN the same list, without self-pairing.
                valid = [w for w in list_lo if w["time"] > dep_time]
                for a in range(len(valid)):
                    for b in range(a + 1, len(valid)):
                        record_match(dep, valid[a], valid[b])
                        if (MAX_LINKS_PER_DEPOSIT is not None
                                and pairs_per_deposit[dep_id] >= MAX_LINKS_PER_DEPOSIT):
                            deposit_truncated[dep_id] = True
                            stop = True
                            break
                    if stop:
                        break
            else:
                # Case a1 != a2: cartesian product between the two lists.
                valid_lo = [w for w in list_lo if w["time"] > dep_time]
                valid_hi = [w for w in list_hi if w["time"] > dep_time]
                for w1 in valid_lo:
                    for w2 in valid_hi:
                        record_match(dep, w1, w2)
                        if (MAX_LINKS_PER_DEPOSIT is not None
                                and pairs_per_deposit[dep_id] >= MAX_LINKS_PER_DEPOSIT):
                            deposit_truncated[dep_id] = True
                            stop = True
                            break
                    if stop:
                        break
            if stop:
                break
            lo += 1
            hi -= 1
        elif diff < 0:
            lo += 1   # sum too small -> raise the small amount
        else:
            hi -= 1   # sum too large -> lower the large amount

# We carry the number of pairs over to each deposit.
# NB: the naive_anonymity_set and s(d) computation is done in a SECOND script
# (compute_s_score.py) so as not to slow down this base program.
for dep in deposits:
    dep["pairs_count"] = pairs_per_deposit[dep["dep_id"]]
    # counter(d): number of distinct withdraws involved in >=1 pair
    dep["linked_withdraws_count"] = len(distinct_withdraws_per_deposit[dep["dep_id"]])
    dep["truncated"] = deposit_truncated[dep["dep_id"]]

# ================================================================
# 6. Terminal results
# ================================================================
print(f"\nNumber of possible linkages (1 deposit = 2 withdraws): {len(matches)}")
n_with_pair = sum(1 for d in deposits if d['pairs_count'] > 0)
n_no_pair = sum(1 for d in deposits if d['pairs_count'] == 0)
n_dep = len(deposits)
pct_with_pair = (100 * n_with_pair / n_dep) if n_dep else 0.0
pct_no_pair = (100 * n_no_pair / n_dep) if n_dep else 0.0
print(f"Deposits with >=1 pair        : {n_with_pair}  ({pct_with_pair:.1f}% of {n_dep} loaded)")
print(f"Deposits with 0 pair          : {n_no_pair}  ({pct_no_pair:.1f}% of {n_dep} loaded)")
print(f"Deposits truncated (cutoff)   : {sum(1 for d in deposits if d['truncated'])}")
print(f"Distinct matching pairs       : {len(deposits_per_pair)}")

# ================================================================
# 7. Write the CSVs
# ================================================================
# 7a. Detailed matches
fieldnames = [
    "dep_hash", "dep_address", "dep_amount",
    "w1_hash", "w1_address", "w1_amount",
    "w2_hash", "w2_address", "w2_amount",
    "difference",
]
with open(MATCHES_OUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(m for m in matches if not m["_excluded"])
print(f"\nMatches saved to           {MATCHES_OUT_PATH}")

# 7b. Pairs-per-deposit counter (zeros included, for histogram)
#     Two files, same format:
#       - DEP_COUNTER_OUT_PATH     : FILTERED (address-reuse excluded), for figures + s(d).
#       - DEP_COUNTER_ALL_OUT_PATH : address-reuse INCLUDED, for total anonymity loss.
dep_fields = [
    "dep_hash", "dep_address", "dep_amount", "dep_time",
    "pairs_count", "linked_withdraws_count", "truncated",
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
                "pairs_count": dep["pairs_count"],
                "linked_withdraws_count": dep["linked_withdraws_count"],
                "truncated": int(dep["truncated"]),
            })


write_per_deposit(DEP_COUNTER_OUT_PATH, include_excluded=False)
print(f"Per-deposit counter (filtered) saved  {DEP_COUNTER_OUT_PATH}")
write_per_deposit(DEP_COUNTER_ALL_OUT_PATH, include_excluded=True)
print(f"Per-deposit counter (ALL)      saved  {DEP_COUNTER_ALL_OUT_PATH}")

# 7c. Deposits-per-withdraw-pair counter
with open(PAIR_COUNTER_OUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["w1_id", "w2_id", "deposits_count"])
    for (w1_id, w2_id), cnt in deposits_per_pair.items():
        writer.writerow([w1_id, w2_id, cnt])
print(f"Per-pair counter saved     {PAIR_COUNTER_OUT_PATH}")