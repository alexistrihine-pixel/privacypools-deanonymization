"""
Privacy Pools — UI Percentage Heuristic
========================================

Hypothesis: users withdraw funds using the preset UI buttons (25%, 50%, 75%,
100%). This script detects on-chain traces of that behaviour and produces
several figures. Two counting philosophies are used and must not be confused:

  * Figures 1 & 2 use CONSUMPTION matching: each withdraw is claimed by at most
    one deposit (chronological order), so a withdraw is counted only once. This
    answers "how many withdraws look like a given button click".

  * Figures 3-6 use EXISTENCE matching: no consumption, we only test whether a
    counterpart exists. This answers "how many deposits/withdraws have matching
    counterparts". Figures 3 & 4 count objects once per button; figures 5 & 6
    count all links per object.

Everywhere the heuristic works in two hops:
  * hop 1 (direct): withdraw = deposit * p
  * hop 2 (chained): a first withdraw W1 = deposit * p1 leaves a remaining
    balance, then W2 = deposit * (1 - p1) * p2.
Figures 3-6 additionally split their bars by hop so hop-1 and hop-2 contributions
are visible on the same bar.

Note: with consumption (figs 1 & 2) the deposit processing order affects the
result; chronological order is used. Existence matching (figs 3-6) is
order-independent, but its per-button columns are not mutually exclusive, so
their sum is not a count of unique objects.
"""

import bisect
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEPOSITS_PATH = "data/processed_privacypools_eth_pool_deposits.csv"
WITHDRAWS_PATH = "data/processed_privacypools_data_withdraws.csv"
OUT_DIR = "figures/Heuristics/UI"

UI_MATCHES_OUT_PATH = "data/Heuristics/UI_Percentage/ui_matches.txt"
UI_NO_REMAINING_BALANCE_DEPOSITORS_OUT_PATH = "data/Heuristics/UI_Percentage/ui_no_balance_depositors.txt"

TARGET_PERCENTAGES = [0.25, 0.50, 0.75, 1.00]
TOLERANCE = 1e-6        # amount matching tolerance (ETH)
MIN_BALANCE = 1e-4      # below this remaining balance a deposit is "empty"

# Consistent hop colours across every figure.
COLOR_HOP1 = "#d9846c"
COLOR_HOP2 = "#7fa8d9"

# Link-count histograms (figs 5 & 6): objects with more than this many links are
# dropped from the histogram so the low-link detail stays readable. Extreme
# objects (round-number amounts shared by thousands of deposits, i.e. likely
# routers/contracts rather than real users) would otherwise stretch the x-axis
# to thousands of bars. Set to None to keep every value. Try 50 / 100 / 200.
LINK_HISTOGRAM_THRESHOLD = 50

# Amount-rarity filter. A deposit amount shared by more than RARITY_THRESHOLD
# deposits is considered "common": on such round amounts the heuristic matches
# almost everything by arithmetic coincidence and carries no de-anonymising
# signal. When RARITY_FILTER_ENABLED is True, deposits on common amounts are
# excluded from the existence-based figures (3-6). The distribution is bimodal
# (unique/rare amounts vs a few massively shared round amounts) so any threshold
# between ~5 and ~50 separates the two groups cleanly.
RARITY_THRESHOLD = 5
RARITY_FILTER_ENABLED = True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load and clean deposits and withdraws CSV files."""
    deposits = pd.read_csv(DEPOSITS_PATH)
    withdraws = pd.read_csv(WITHDRAWS_PATH)

    # Timestamps use a European comma decimal separator, hence the replace.
    deposits['time'] = pd.to_datetime(
        deposits['time'].str.replace(',', '.'), errors='coerce', utc=True)
    withdraws['evt_block_time'] = pd.to_datetime(
        withdraws['evt_block_time'].str.replace(',', '.'), errors='coerce', utc=True)

    # Tag each deposit with how many deposits share its amount (its "frequency").
    # Computed on the full set before any filtering so rarity reflects reality.
    freq = deposits['amount_eth_net'].round(9).map(
        deposits['amount_eth_net'].round(9).value_counts())
    deposits['amount_freq'] = freq.to_numpy()

    # Deposits are processed in chronological order so that a deposit can only
    # claim a later withdraw and older deposits get first pick.
    deposits = deposits.sort_values('time', kind='mergesort').reset_index(drop=True)
    return deposits, withdraws


def filter_common_amounts(deposits):
    """
    Keep only deposits whose amount is rare (shared by <= RARITY_THRESHOLD
    deposits). On common round amounts the heuristic matches almost everything by
    coincidence, so those deposits are dropped from the existence-based figures.
    Returns the filtered frame and the number of deposits removed.
    """
    if not RARITY_FILTER_ENABLED:
        return deposits, 0
    mask = deposits['amount_freq'] <= RARITY_THRESHOLD
    return deposits[mask].copy(), int((~mask).sum())


# ---------------------------------------------------------------------------
# Sorted-amount index (fast O(log n) lookups via bisect)
# ---------------------------------------------------------------------------
class SortedAmountIndex:
    """
    Keeps rows sorted by amount, with aligned timestamps, to answer amount+time
    queries quickly. Optionally supports single-use "consumption".
    """

    def __init__(self, amounts, times, track_consumed=False):
        order = np.argsort(amounts, kind='mergesort')
        self.amounts = amounts[order]
        self.times = times[order]
        self.consumed = (np.zeros(len(order), dtype=bool)
                         if track_consumed else None)

    def _range(self, amount):
        """Index range of rows whose amount ~= `amount` (within TOLERANCE)."""
        lo = bisect.bisect_left(self.amounts, amount - TOLERANCE)
        hi = bisect.bisect_right(self.amounts, amount + TOLERANCE)
        return lo, hi

    def times_after(self, amount, after_time):
        """Timestamps of rows with matching amount and time >= after_time."""
        lo, hi = self._range(amount)
        return [self.times[k] for k in range(lo, hi) if self.times[k] >= after_time]

    def count_after(self, amount, after_time):
        """Number of rows with matching amount and time >= after_time."""
        lo, hi = self._range(amount)
        return sum(1 for k in range(lo, hi) if self.times[k] >= after_time)

    def exists_before(self, amount, before_time):
        """True if a row exists with matching amount and time <= before_time."""
        lo, hi = self._range(amount)
        return any(self.times[k] <= before_time for k in range(lo, hi))

    def count_before(self, amount, before_time):
        """Number of rows with matching amount and time <= before_time."""
        lo, hi = self._range(amount)
        return sum(1 for k in range(lo, hi) if self.times[k] <= before_time)

    def times_before(self, amount, before_time):
        """Timestamps of rows with matching amount and time <= before_time."""
        lo, hi = self._range(amount)
        return [self.times[k] for k in range(lo, hi) if self.times[k] <= before_time]

    def exists_between(self, amount, start_time, end_time):
        """True if a row exists with matching amount and start <= time <= end."""
        lo, hi = self._range(amount)
        return any(start_time <= self.times[k] <= end_time for k in range(lo, hi))

    def claim_earliest_after(self, amount, after_time):
        """
        Claim (consume) the earliest not-yet-consumed row with matching amount
        and time >= after_time. Return its timestamp, or None if none is free.
        """
        lo, hi = self._range(amount)
        best_k, best_t = -1, None
        for k in range(lo, hi):
            if self.consumed[k] or self.times[k] < after_time:
                continue
            if best_t is None or self.times[k] < best_t:
                best_t, best_k = self.times[k], k
        if best_k == -1:
            return None
        self.consumed[best_k] = True
        return best_t

    @property
    def n_consumed(self):
        return int(self.consumed.sum()) if self.consumed is not None else 0


# ---------------------------------------------------------------------------
# Analysis 1 — consumption matching (feeds figures 1 & 2)
# ---------------------------------------------------------------------------
def run_consumption_matching(deposits, w_index):
    """
    For each deposit (chronological), claim one withdraw per button in 1-hop,
    then chain a 2-hop match on the updated balance. Each withdraw is consumed
    once. Returns per-button hit counts, per-deposit match counts, and the
    depositor lists needed for the heuristic union.
    """
    percentage_hits = {p: 0 for p in TARGET_PERCENTAGES}
    per_deposit_counts = []
    matching_depositors = []
    zero_balance_depositors = []

    for i in deposits.index:
        net = deposits.at[i, 'amount_eth_net']
        deposit_time = deposits.at[i, 'time']
        depositor = deposits.at[i, 'depositor']
        cnt = 0

        if pd.isna(deposit_time) or net <= MIN_BALANCE:
            per_deposit_counts.append(0)
            continue

        for p1 in TARGET_PERCENTAGES:
            amount_w1 = net * p1
            if amount_w1 <= TOLERANCE:
                continue

            # 1-hop: claim one withdraw for this deposit at p1.
            w1_time = w_index.claim_earliest_after(amount_w1, deposit_time)
            if w1_time is None:
                continue
            percentage_hits[p1] += 1
            cnt += 1
            if p1 == 1.00:
                zero_balance_depositors.append(depositor)

            # 2-hop: only if some balance remains after the first withdraw.
            remaining = net * (1 - p1)
            if remaining <= MIN_BALANCE:
                continue
            for p2 in TARGET_PERCENTAGES:
                amount_w2 = remaining * p2
                if amount_w2 <= TOLERANCE:
                    continue
                w2_time = w_index.claim_earliest_after(amount_w2, w1_time)
                if w2_time is None:
                    continue
                percentage_hits[p2] += 1
                cnt += 1
                if p2 == 1.00:
                    zero_balance_depositors.append(depositor)

        per_deposit_counts.append(cnt)
        if cnt > 0:
            matching_depositors.append(depositor)

    return (percentage_hits, per_deposit_counts,
            matching_depositors, zero_balance_depositors)


# ---------------------------------------------------------------------------
# Analysis 2 — existence matching per button, split by hop (figures 3 & 4)
# ---------------------------------------------------------------------------
def count_deposits_per_button(deposits, w_index):
    """
    Count deposits with at least one matching withdraw per button, each deposit
    counted once per button. Split by hop: a deposit is attributed to hop 1 for
    a button if it matches there, otherwise to hop 2. No consumption.
    Returns (hop1_counts, hop2_counts).
    """
    hop1 = {p: 0 for p in TARGET_PERCENTAGES}
    hop2 = {p: 0 for p in TARGET_PERCENTAGES}

    for i in deposits.index:
        net = deposits.at[i, 'amount_eth_net']
        deposit_time = deposits.at[i, 'time']
        if pd.isna(deposit_time) or net <= MIN_BALANCE:
            continue

        touched_h1 = {p: False for p in TARGET_PERCENTAGES}
        touched_h2 = {p: False for p in TARGET_PERCENTAGES}

        for p1 in TARGET_PERCENTAGES:
            amount_w1 = net * p1
            if amount_w1 <= TOLERANCE:
                continue
            t1_list = w_index.times_after(amount_w1, deposit_time)
            if not t1_list:
                continue
            touched_h1[p1] = True

            remaining = net * (1 - p1)
            if remaining <= MIN_BALANCE:
                continue
            earliest_t1 = min(t1_list)
            for p2 in TARGET_PERCENTAGES:
                amount_w2 = remaining * p2
                if amount_w2 <= TOLERANCE:
                    continue
                if w_index.times_after(amount_w2, earliest_t1):
                    touched_h2[p2] = True

        for p in TARGET_PERCENTAGES:
            if touched_h1[p]:
                hop1[p] += 1
            elif touched_h2[p]:
                hop2[p] += 1
    return hop1, hop2


def count_withdraws_per_button(withdraws, d_index, w_index):
    """
    Count withdraws with at least one matching deposit per button, each withdraw
    counted once per button, split by hop (symmetric to the deposit version).
    Returns (hop1_counts, hop2_counts).
    """
    hop1 = {p: 0 for p in TARGET_PERCENTAGES}
    hop2 = {p: 0 for p in TARGET_PERCENTAGES}

    for i in withdraws.index:
        w_amt = withdraws.at[i, 'eth_amount']
        w_time = withdraws.at[i, 'evt_block_time']
        if pd.isna(w_time) or w_amt <= MIN_BALANCE:
            continue

        touched_h1 = {p: False for p in TARGET_PERCENTAGES}
        touched_h2 = {p: False for p in TARGET_PERCENTAGES}

        # 1-hop: deposit = withdraw / p, before the withdraw.
        for p in TARGET_PERCENTAGES:
            if d_index.exists_before(w_amt / p, w_time):
                touched_h1[p] = True

        # 2-hop: withdraw = D * (1 - p1) * p2, with an intermediate W1 = D * p1.
        for p1 in TARGET_PERCENTAGES:
            if p1 == 1.00:
                continue
            for p2 in TARGET_PERCENTAGES:
                if touched_h2[p2]:
                    continue
                denom = (1 - p1) * p2
                if denom <= 0:
                    continue
                deposit_amount = w_amt / denom
                if deposit_amount <= MIN_BALANCE:
                    continue
                for d_time in d_index.times_before(deposit_amount, w_time):
                    if w_index.exists_between(deposit_amount * p1, d_time, w_time):
                        touched_h2[p2] = True
                        break

        for p in TARGET_PERCENTAGES:
            if touched_h1[p]:
                hop1[p] += 1
            elif touched_h2[p]:
                hop2[p] += 1
    return hop1, hop2


# ---------------------------------------------------------------------------
# Analysis 3 — link counts per object, split by hop (figures 5 & 6)
# ---------------------------------------------------------------------------
def links_per_deposit(deposits, w_index):
    """
    For each deposit, count linked withdraws split into (hop1, hop2). All links,
    every button class, no consumption. Returns a list of (hop1, hop2) tuples
    aligned with deposits.index.
    """
    result = []
    for i in deposits.index:
        net = deposits.at[i, 'amount_eth_net']
        deposit_time = deposits.at[i, 'time']
        if pd.isna(deposit_time) or net <= MIN_BALANCE:
            result.append((0, 0))
            continue

        c1 = c2 = 0
        for p1 in TARGET_PERCENTAGES:
            amount_w1 = net * p1
            if amount_w1 <= TOLERANCE:
                continue
            t1_list = w_index.times_after(amount_w1, deposit_time)
            c1 += len(t1_list)                       # all 1-hop matches

            remaining = net * (1 - p1)
            if remaining <= MIN_BALANCE or not t1_list:
                continue
            earliest_t1 = min(t1_list)
            for p2 in TARGET_PERCENTAGES:
                amount_w2 = remaining * p2
                if amount_w2 <= TOLERANCE:
                    continue
                c2 += w_index.count_after(amount_w2, earliest_t1)  # all 2-hop
        result.append((c1, c2))
    return result


def links_per_withdraw(withdraws, d_index, w_index):
    """
    For each withdraw, count linked deposits split into (hop1, hop2). All links,
    every button class, no consumption, deposit precedes withdraw. Returns a list
    of (hop1, hop2) tuples aligned with withdraws.index.
    """
    result = []
    for i in withdraws.index:
        w_amt = withdraws.at[i, 'eth_amount']
        w_time = withdraws.at[i, 'evt_block_time']
        if pd.isna(w_time) or w_amt <= MIN_BALANCE:
            result.append((0, 0))
            continue

        c1 = c2 = 0
        # 1-hop: deposit = withdraw / p, before the withdraw.
        for p in TARGET_PERCENTAGES:
            c1 += d_index.count_before(w_amt / p, w_time)

        # 2-hop: withdraw = D * (1 - p1) * p2, with intermediate W1 = D * p1.
        for p1 in TARGET_PERCENTAGES:
            if p1 == 1.00:
                continue
            for p2 in TARGET_PERCENTAGES:
                denom = (1 - p1) * p2
                if denom <= 0:
                    continue
                deposit_amount = w_amt / denom
                if deposit_amount <= MIN_BALANCE:
                    continue
                for d_time in d_index.times_before(deposit_amount, w_time):
                    if w_index.exists_between(deposit_amount * p1, d_time, w_time):
                        c2 += 1
        result.append((c1, c2))
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_depositor_lists(matching_depositors, zero_balance_depositors):
    """Persist unique depositor address lists for later heuristic unions."""
    unique_matching = list(set(matching_depositors))
    unique_zero_balance = list(set(zero_balance_depositors))

    with open(UI_MATCHES_OUT_PATH, "w") as f:
        for addr in unique_matching:
            f.write(f"{addr}\n")
    print(f"Saved {len(unique_matching)} addresses to {UI_MATCHES_OUT_PATH}")

    with open(UI_NO_REMAINING_BALANCE_DEPOSITORS_OUT_PATH, "w") as f:
        for addr in unique_zero_balance:
            f.write(f"{addr}\n")
    print(f"Saved {len(unique_zero_balance)} addresses to "
          f"{UI_NO_REMAINING_BALANCE_DEPOSITORS_OUT_PATH}")


def print_withdraw_matches(withdraws, wd_links, top=None):
    """
    Print, for manual verification, every withdraw that has at least one linked
    deposit, sorted by total links descending: total / hop1 / hop2 / index /
    eth_amount / tx_hash.
    """
    rows = []
    for pos, i in enumerate(withdraws.index):
        c1, c2 = wd_links[pos]
        if c1 + c2 > 0:
            rows.append((c1 + c2, c1, c2, i,
                         withdraws.at[i, 'eth_amount'],
                         withdraws.at[i, 'tx_hash']))
    rows.sort(reverse=True)

    print(f"\n=== Withdraws with at least one linked deposit "
          f"({len(rows)} total) ===")
    print(f"{'total':>6} {'hop1':>6} {'hop2':>6} {'idx':>6} "
          f"{'eth_amount':>16}  tx_hash")
    shown = rows if top is None else rows[:top]
    for total, c1, c2, i, amt, txh in shown:
        print(f"{total:>6} {c1:>6} {c2:>6} {i:>6} {amt:>16.9f}  {txh}")


# --- Simple bar charts (figures 1 & 2, unchanged) ---
def bar_figure(labels, values, color, title, xlabel, ylabel, out_path):
    """Generic single-series bar chart with value labels on top."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=color, edgecolor="black", width=0.5)
    ax.bar_label(bars, padding=3, fontsize=10, fontweight="bold")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ymargin(0.2)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)


def plot_button_hits(percentage_hits):
    """Figure 1 — consumed withdraws per button (1-hop + 2-hop)."""
    labels = [f"{int(p * 100)}%" for p in TARGET_PERCENTAGES]
    values = [percentage_hits[p] for p in TARGET_PERCENTAGES]
    bar_figure(
        labels, values, "#4a47f8",
        "Number of occurrences per target percentage (1-hop + 2-hop)",
        "Percentage button selected in the UI (25%, 50%, 75%, 100%)",
        "Number of matches",
        f"{OUT_DIR}/percentage_hits_distribution.png")


def plot_matches_per_deposit(per_deposit_counts):
    """Figure 2 — histogram of the number of matches per deposit (log Y)."""
    counter = Counter(per_deposit_counts)
    x_values = sorted(counter.keys())
    y_values = [counter[x] for x in x_values]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x_values, y_values, color="skyblue", edgecolor="black", width=0.6)
    ax.set_yscale("log")
    ax.bar_label(bars, padding=2, fontsize=7)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values, rotation=30, fontsize=8)
    ax.set_title("UI Heuristic Match Occurrences (Log Y-Scale)")
    ax.set_xlabel("Number of matches")
    ax.set_ylabel("Occurrences (Log Scale)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/UI_percentage_heuristic_recursive_twice.png")


# --- Per-button stacked bars by hop (figures 3 & 4) ---
def stacked_button_figure(hop1, hop2, color1, color2, title, xlabel, ylabel, out_path):
    """Per-button bar chart, hop1 at the bottom and hop2 stacked on top."""
    labels = [f"{int(p * 100)}%" for p in TARGET_PERCENTAGES]
    y1 = [hop1[p] for p in TARGET_PERCENTAGES]
    y2 = [hop2[p] for p in TARGET_PERCENTAGES]

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(labels, y1, color=color1, edgecolor="black", width=0.5,
                label="Hop 1")
    b2 = ax.bar(labels, y2, bottom=y1, color=color2, edgecolor="black", width=0.5,
                label="Hop 2")
    # Total label on top of each stacked bar.
    for x, (a, b) in enumerate(zip(y1, y2)):
        ax.text(x, a + b, str(a + b), ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ymargin(0.2)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)


def plot_deposits_per_button(hop1, hop2):
    """Figure 3 — deposits per button, split by hop."""
    stacked_button_figure(
        hop1, hop2, COLOR_HOP1, COLOR_HOP2,
        "Number of deposits with at least one matching withdraw per button",
        "Percentage button (25%, 50%, 75%, 100%) — each deposit counted once per button",
        "Number of deposits",
        f"{OUT_DIR}/deposits_per_button_distribution.png")


def plot_withdraws_per_button(hop1, hop2):
    """Figure 4 — withdraws per button, split by hop."""
    stacked_button_figure(
        hop1, hop2, COLOR_HOP1, COLOR_HOP2,
        "Number of withdraws with at least one matching deposit per button",
        "Percentage button (25%, 50%, 75%, 100%) — each withdraw counted once per button",
        "Number of withdraws",
        f"{OUT_DIR}/withdraws_per_button_distribution.png")


# --- Link-count histograms, stacked by hop (figures 5 & 6) ---
def stacked_link_histogram(pairs, title, xlabel, ylabel, out_path):
    """
    Histogram of total links per object (x = total link count), one bar per exact
    value, split into objects whose links are hop-1 only vs objects that include
    at least one hop-2 link. Objects with more than LINK_HISTOGRAM_THRESHOLD
    links are dropped so the low-link detail stays readable (see the constant's
    comment). Bar height counts objects; colour indicates the presence of hop-2
    links, not their volume (read the scatter plot for the real volume).
    """
    threshold = LINK_HISTOGRAM_THRESHOLD
    kept = [(a, b) for a, b in pairs
            if threshold is None or (a + b) <= threshold]
    skipped = len(pairs) - len(kept)

    totals = [a + b for a, b in kept]
    includes_h2 = [b > 0 for a, b in kept]

    max_total = max(totals) if totals else 0
    x_values = list(range(0, max_total + 1))
    only_h1 = {x: 0 for x in x_values}
    with_h2 = {x: 0 for x in x_values}
    for total, has_h2 in zip(totals, includes_h2):
        if has_h2:
            with_h2[total] += 1
        else:
            only_h1[total] += 1

    y1 = [only_h1[x] for x in x_values]
    bar_totals = [only_h1[x] + with_h2[x] for x in x_values]
    pos = range(len(x_values))

    # Overlay technique for the log axis: draw the full total in the hop-2
    # colour, then overlay the hop-1-only part on top (a stacked bottom=0 bar
    # renders wrong on a log scale).
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(pos, bar_totals, color=COLOR_HOP2, edgecolor="black", width=0.85,
           label="Includes hop 2")
    ax.bar(pos, y1, color=COLOR_HOP1, edgecolor="black", width=0.85,
           label="Hop 1 only")
    ax.set_yscale("log")
    if any(bar_totals):
        ax.set_ylim(top=max(bar_totals) * 3)

    # Count label on top of every non-empty bar.
    for x, total in zip(pos, bar_totals):
        if total > 0:
            ax.text(x, total * 1.15, str(total), ha="center", va="bottom",
                    fontsize=7)

    ax.set_xticks(list(pos))
    ax.set_xticklabels(x_values, rotation=90, fontsize=7)
    full_title = title
    if threshold is not None:
        full_title += f"  (threshold {threshold}, {skipped} objects dropped)"
    ax.set_title(full_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)


def plot_links_per_deposit_hist(dep_links):
    """Figure 5 — histogram of linked withdraws per deposit, stacked by hop."""
    stacked_link_histogram(
        dep_links,
        "Linked withdraws per deposit — all UI classes (hop 1 / hop 2, Log Y)",
        "Number of linked withdraws",
        "Number of deposits (Log Scale)",
        f"{OUT_DIR}/links_per_deposit_histogram.png")


def plot_links_per_withdraw_hist(wd_links):
    """Figure 6 — histogram of linked deposits per withdraw, stacked by hop."""
    stacked_link_histogram(
        wd_links,
        "Linked deposits per withdraw — all UI classes (hop 1 / hop 2, Log Y)",
        "Number of linked deposits",
        "Number of withdraws (Log Scale)",
        f"{OUT_DIR}/links_per_withdraw_histogram.png")


# --- Scatter plots (per-object link counts, hop1 vs hop2) ---
def scatter_links(pairs, title, ylabel, out_path):
    """Scatter of per-object link counts: hop1 and hop2 as two coloured series."""
    h1 = np.array([a for a, b in pairs])
    h2 = np.array([b for a, b in pairs])
    idx = np.arange(len(pairs))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.scatter(idx, h1, s=8, c=COLOR_HOP1, alpha=0.6, label="Hop 1")
    ax.scatter(idx, h2, s=8, c=COLOR_HOP2, alpha=0.6, label="Hop 2")
    ax.set_title(title)
    ax.set_xlabel("Object index")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)


def plot_scatter_deposit(dep_links):
    """Scatter for figure 5 data."""
    scatter_links(
        dep_links,
        "Linked withdraws per deposit — scatter (hop 1 / hop 2)",
        "Number of linked withdraws",
        f"{OUT_DIR}/links_per_deposit_scatter.png")


def plot_scatter_withdraw(wd_links):
    """Scatter for figure 6 data."""
    scatter_links(
        wd_links,
        "Linked deposits per withdraw — scatter (hop 1 / hop 2)",
        "Number of linked deposits",
        f"{OUT_DIR}/links_per_withdraw_scatter.png")


# ---------------------------------------------------------------------------
# Analysis 4 — heuristic power vs amount rarity (figures 7 & 8)
# ---------------------------------------------------------------------------
def links_vs_rarity(deposits, w_index):
    """
    For every deposit (unfiltered), return (amount_frequency, n_links) where
    n_links is the number of 1-hop linked withdraws. Shows how the heuristic's
    output scales with how common the deposit amount is.
    """
    freqs = []
    counts = []
    for i in deposits.index:
        net = deposits.at[i, 'amount_eth_net']
        deposit_time = deposits.at[i, 'time']
        if pd.isna(deposit_time) or net <= MIN_BALANCE:
            continue
        c = 0
        for p in TARGET_PERCENTAGES:
            amount = net * p
            if amount <= TOLERANCE:
                continue
            c += w_index.count_after(amount, deposit_time)
        freqs.append(int(deposits.at[i, 'amount_freq']))
        counts.append(c)
    return np.array(freqs), np.array(counts)


def plot_rarity_scatter(freqs, counts):
    """Figure 7 — amount frequency vs number of linked withdraws (symlog)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(freqs, counts, s=10, alpha=0.4, c="#4a47f8")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_xlabel("Deposit amount frequency (deposits sharing this amount)")
    ax.set_ylabel("Number of linked withdraws (1-hop)")
    ax.set_title("Heuristic power collapses on common amounts")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/rarity_vs_links_scatter.png")


def plot_rarity_bars(freqs, counts):
    """Figure 8 — mean linked withdraws per amount-frequency class."""
    classes = [(1, 1, "unique"), (2, 5, "2-5"), (6, 50, "6-50"),
               (51, 10 ** 12, ">50")]
    labels, means, colors = [], [], ["#2ca02c", "#8bc34a", "#ff9800", "#d62728"]
    for lo, hi, lab in classes:
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() == 0:
            continue
        labels.append(f"{lab}\n(n={int(mask.sum())})")
        means.append(counts[mask].mean())

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, means, color=colors[:len(labels)], edgecolor="black")
    ax.bar_label(bars, fmt="%.1f", fontweight="bold", padding=3)
    ax.set_ylabel("Mean number of linked withdraws per deposit")
    ax.set_xlabel("Deposit amount frequency class")
    ax.set_title("Mean heuristic links by amount rarity")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/rarity_vs_links_bars.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    deposits, withdraws = load_data()

    # Rarity-filtered deposits: on common round amounts the heuristic matches
    # almost everything by coincidence, so those deposits are excluded from the
    # existence-based figures (3-6). Figures 1-2 and the rarity figures (7-8) use
    # the full set.
    deposits_f, n_common = filter_common_amounts(deposits)
    if RARITY_FILTER_ENABLED:
        print(f"Rarity filter ON (threshold {RARITY_THRESHOLD}): "
              f"{n_common} common-amount deposits dropped, "
              f"{len(deposits_f)} kept for figs 3-6.")

    # Deposit index over the RARE deposits, used by every existence-based figure
    # (both the deposit-side ones and the withdraw-side ones, for consistency).
    d_valid = deposits_f.dropna(subset=['time'])
    d_index = SortedAmountIndex(
        d_valid['amount_eth_net'].to_numpy(),
        d_valid['time'].to_numpy())

    # Two withdraw indices, on purpose:
    #   * consumable one, single-use, for the consumption matching (figs 1 & 2)
    #   * fresh non-consumed one, for every existence-based count (figs 3-6)
    w_amounts = withdraws.dropna(subset=['evt_block_time'])['eth_amount'].to_numpy()
    w_times = withdraws.dropna(subset=['evt_block_time'])['evt_block_time'].to_numpy()
    w_index_consumable = SortedAmountIndex(w_amounts, w_times, track_consumed=True)
    w_index = SortedAmountIndex(w_amounts, w_times)

    # --- Analysis 1: consumption matching (figures 1 & 2) ---
    # On the FULL deposit set: consumption already prevents over-counting.
    (percentage_hits, per_deposit_counts,
     matching_depositors, zero_balance_depositors) = run_consumption_matching(
        deposits, w_index_consumable)

    print("percentage_hits   :", percentage_hits)
    print("Button sum        :", sum(percentage_hits.values()))
    print("Consumed withdraws:", w_index_consumable.n_consumed)
    print("Total withdraws   :", len(withdraws))
    assert sum(percentage_hits.values()) == w_index_consumable.n_consumed <= len(withdraws)

    save_depositor_lists(matching_depositors, zero_balance_depositors)

    # --- Analysis 2: existence matching per button, split by hop (figs 3 & 4) ---
    dep_btn_h1, dep_btn_h2 = count_deposits_per_button(deposits_f, w_index)
    wd_btn_h1, wd_btn_h2 = count_withdraws_per_button(withdraws, d_index, w_index)
    print("Deposits per button  hop1:", dep_btn_h1, "hop2:", dep_btn_h2)
    print("Withdraws per button hop1:", wd_btn_h1, "hop2:", wd_btn_h2)

    # --- Analysis 3: link counts per object, split by hop (figs 5 & 6) ---
    dep_links = links_per_deposit(deposits_f, w_index)
    wd_links = links_per_withdraw(withdraws, d_index, w_index)
    print("Deposit links  hop1:", sum(a for a, b in dep_links),
          "hop2:", sum(b for a, b in dep_links))
    print("Withdraw links hop1:", sum(a for a, b in wd_links),
          "hop2:", sum(b for a, b in wd_links))

    # Manual-verification dump for figure 6 (withdraws with linked deposits).
    print_withdraw_matches(withdraws, wd_links)

    # --- Analysis 4: heuristic power vs amount rarity (figs 7 & 8) ---
    # Computed on the FULL deposit set (unfiltered) to show the whole gradient.
    rarity_freqs, rarity_counts = links_vs_rarity(deposits, w_index)

    # --- Figures ---
    plot_button_hits(percentage_hits)
    plot_matches_per_deposit(per_deposit_counts)
    plot_deposits_per_button(dep_btn_h1, dep_btn_h2)
    plot_withdraws_per_button(wd_btn_h1, wd_btn_h2)
    plot_links_per_deposit_hist(dep_links)
    plot_links_per_withdraw_hist(wd_links)
    plot_scatter_deposit(dep_links)
    plot_scatter_withdraw(wd_links)
    plot_rarity_scatter(rarity_freqs, rarity_counts)
    plot_rarity_bars(rarity_freqs, rarity_counts)

    plt.show()


if __name__ == "__main__":
    main()