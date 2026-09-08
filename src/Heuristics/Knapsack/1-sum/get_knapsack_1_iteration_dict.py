import csv
from datetime import datetime, timezone
from decimal import Decimal

# ---- Paths to input data (edit if your files live elsewhere) ----
DEPOSITS_CSV = "data/processed_privacypools_eth_pool_deposits.csv"
WITHDRAWS_CSV = "data/processed_privacypools_data_withdraws.csv"
NO_BALANCE_DEPOSITORS_PATH = "data/Heuristics/Address_Reuse/no_balance_depositors.txt"
MATCHES_OUT_PATH = "data/Heuristics/Knapsack/1-sum/knapsack_1_withdraw_matches.csv"
# Counter outputs (for the histograms)
DEP_COUNTER_OUT_PATH = "data/Heuristics/Knapsack/1-sum/knapsack_1_withdraw_matches_per_deposit.csv"
DEP_COUNTER_ALL_OUT_PATH = "data/Heuristics/Knapsack/1-sum/knapsack_1_withdraw_matches_per_deposit_all.csv"
W_COUNTER_OUT_PATH = "data/Heuristics/Knapsack/1-sum/knapsack_1_withdraw_deposits_per_withdraw.csv"

# ---- Config ----
TOLERANCE = 1e-9          # tight tolerance
MIN_DEPOSIT_HW = 2        # keep only deposits with a "complex" amount
# Cutoff: as soon as a deposit reaches this number of matches, we drop it and
# move on to the next (consistent with the 2-sum and 3-sum). Useful at HW=0
# where round amounts have hundreds of withdraws of the same amount. Set to
# None to disable (original behavior, no limit).
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
        amount_net = float(row["amount_eth_net"])
        if MIN_DEPOSIT_HW > 0 and decimal_fraction_hamming_weight(amount_net) < MIN_DEPOSIT_HW:
            continue
        # Load ALL deposits (HW-filtered), including those excluded by
        # address-reuse. The "excluded" flag then lets us write two files:
        # one filtered (without address-reuse), the other complete.
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
print(f"Deposits address-reuse excluded (kept for '_all' file only): {n_excluded}")
 
print(f"Deposits total in file        : {n_total_dep}")
print(f"Deposits loaded (HW>={MIN_DEPOSIT_HW}, address-reuse included): {len(deposits)}")
print(f"  of which kept (address-reuse excluded): {len(deposits) - n_excluded}")
 
# ================================================================
# 3. Load the withdraws and INDEX by amount
#
#    Prior aggregation by tx_hash (one on-chain transaction can emit several
#    outputs to the same address -> we sum them into a single withdraw).
#    withdraws_by_amount : key = rounded_amount (integer 1e-9)
#                          value = list of withdraws with that amount
# ================================================================
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
 
withdraws_by_amount = {}
n_withdraws = 0
for h, agg in raw_by_hash.items():
    key = q(agg["amount"])
    tx = {
        "w_id": n_withdraws,
        "amount": float(agg["amount"]),
        "time": agg["time"],
        "recipient": agg["recipient"],
        "tx_hash": h,
    }
    withdraws_by_amount.setdefault(key, []).append(tx)
    n_withdraws += 1
 
print(f"Withdraws indexed (aggregated): {n_withdraws}")
print(f"Distinct withdraw amounts     : {len(withdraws_by_amount)}")
 
# ================================================================
# 3b. Initialize the counters (zeros included)
#
#   matches_per_deposit : dep_id -> number of withdraws exactly matching this deposit.
#   deposits_per_withdraw : w_id -> number of distinct deposits this withdraw
#                           can explain ("withdraw-side" counter).
# ================================================================
matches_per_deposit = {dep["dep_id"]: 0 for dep in deposits}
deposits_per_withdraw = {}
# deposit_truncated : dep_id -> True if the deposit reached the cutoff
deposit_truncated = {dep["dep_id"]: False for dep in deposits}
 
# ================================================================
# 4. 1-sum search: deposit == withdraw (same amount)
#
#    For each deposit, we directly fetch withdraws_by_amount[dep_q]
#    (O(1) lookup) and keep the withdraws LATER than the deposit.
#    No combinatorics here: this is the simplest case.
# ================================================================
matches = []
tol_q = q(TOLERANCE)
 
 
def record_match(dep, w):
    diff_dec = abs(Decimal(str(w["amount"])) - Decimal(str(dep["amount_net"])))
    matches.append({
        "dep_hash": dep["tx_hash"], "dep_address": dep["depositor"], "dep_amount": dep["amount_net"],
        "w1_hash": w["tx_hash"], "w1_address": w["recipient"], "w1_amount": w["amount"],
        "difference": float(diff_dec),
        "_excluded": dep["excluded"],   # interne : sert a filtrer a l'ecriture
    })
    matches_per_deposit[dep["dep_id"]] += 1
    # The per-withdraw counter only concerns the filtered population (it feeds
    # figure 5 on the withdraw side): we ignore address-reuse deposits here.
    if not dep["excluded"]:
        deposits_per_withdraw[w["w_id"]] = deposits_per_withdraw.get(w["w_id"], 0) + 1
 
 
for dep in deposits:
    dep_q = dep["amount_q"]
    dep_time = dep["time"]
    dep_id = dep["dep_id"]
    stop = False   # becomes True when the cutoff is reached
 
    # Amount exactly equal (within tolerance: we test dep_q and its immediate neighbors)
    for k in (dep_q - tol_q, dep_q, dep_q + tol_q):
        bucket = withdraws_by_amount.get(k)
        if not bucket:
            continue
        for w in bucket:
            if w["time"] > dep_time:
                record_match(dep, w)
                if (MAX_LINKS_PER_DEPOSIT is not None
                        and matches_per_deposit[dep_id] >= MAX_LINKS_PER_DEPOSIT):
                    deposit_truncated[dep_id] = True
                    stop = True
                    break
        if stop:
            break
 
# Carry the counter over to each deposit
for dep in deposits:
    dep["matches_count"] = matches_per_deposit[dep["dep_id"]]
    dep["truncated"] = deposit_truncated[dep["dep_id"]]
 
# ================================================================
# 5. Terminal results
# ================================================================
print(f"\nTotal 1-sum matches (deposit == 1 withdraw): {len(matches)}")
n_kept = len(deposits) - n_excluded
n_match_filtered = sum(1 for d in deposits if not d['excluded'] and d['matches_count'] > 0)
n_match_all = sum(1 for d in deposits if d['matches_count'] > 0)
pct_filtered = (100 * n_match_filtered / n_kept) if n_kept else 0.0
pct_all = (100 * n_match_all / len(deposits)) if deposits else 0.0
print(f"Deposits with >=1 match (filtered)      : {n_match_filtered}  ({pct_filtered:.1f}% of {n_kept} kept)")
print(f"Deposits with >=1 match (all)           : {n_match_all}  ({pct_all:.1f}% of {len(deposits)} loaded)")
print(f"Deposits truncated (cutoff)             : {sum(1 for d in deposits if d['truncated'])}")
 
# ================================================================
# 6. Write the CSVs
# ================================================================
# 6a. Detailed matches (filtered population: address-reuse excluded)
fieldnames = [
    "dep_hash", "dep_address", "dep_amount",
    "w1_hash", "w1_address", "w1_amount",
    "difference",
]
with open(MATCHES_OUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(m for m in matches if not m["_excluded"])
print(f"\nMatches saved to           {MATCHES_OUT_PATH}")
 
 
# 6b. Matches-per-deposit counter
#     Two files, same format:
#       - DEP_COUNTER_OUT_PATH     : FILTERED population (address-reuse excluded).
#         Feeds the figures and the 2-sum/3-sum filter.
#       - DEP_COUNTER_ALL_OUT_PATH : ALL deposits (address-reuse included).
#         To know the total 1-sum coverage.
dep_fields = [
    "dep_hash", "dep_address", "dep_amount", "dep_time",
    "matches_count", "truncated",
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
                "matches_count": dep["matches_count"],
                "truncated": int(dep["truncated"]),
            })
 
 
write_per_deposit(DEP_COUNTER_OUT_PATH, include_excluded=False)
print(f"Per-deposit counter (filtered) saved  {DEP_COUNTER_OUT_PATH}")
write_per_deposit(DEP_COUNTER_ALL_OUT_PATH, include_excluded=True)
print(f"Per-deposit counter (ALL)      saved  {DEP_COUNTER_ALL_OUT_PATH}")
 
# 6c. Deposits-per-withdraw counter (filtered population, indexed by w_id)
with open(W_COUNTER_OUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["w_id", "deposits_count"])
    for w_id, cnt in deposits_per_withdraw.items():
        writer.writerow([w_id, cnt])
print(f"Per-withdraw counter saved {W_COUNTER_OUT_PATH}")