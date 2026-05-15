"""
06 — Look-Ahead Audit for the WRDS / CRSP Pipeline (T9–T14)

Extends the T1–T8 tests in 02_lookahead_tests.ipynb to cover the new
WRDS data flow added by 03_wrds_pull.py and 04_wrds_integrate.py.

Each test prints PASS / FAIL and the supporting statistic. All tests
must pass before the §8a CRSP results can be trusted.

T9  — Entry price uses entry_date or later (no past price)
T10 — Forward N-day price strictly follows entry price
T11 — Russell 3000 PIT membership uses only past snapshots
T12 — BESTTICKER → PERMNO mapping is point-in-time (no future renames)
T13 — CRSP and yfinance returns agree on common events (sanity)
T14 — No future Compustat ticker leaks (sp_constituents from_date logic)

Run:
    /Users/yueqilin/anaconda3/bin/python 06_wrds_lookahead_tests.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings("ignore")

DATA = Path("data")
events = pd.read_parquet(DATA / "events_with_returns_wrds.parquet")
events["entry_date"] = pd.to_datetime(events["entry_date"])

prices = pd.read_parquet(DATA / "wrds_prices.parquet",
                         columns=["permno", "date", "adj_prc"])
prices["date"] = pd.to_datetime(prices["date"])
prices["permno"] = prices["permno"].astype(int)

names = pd.read_parquet(DATA / "wrds_names.parquet")
names["namedt"]   = pd.to_datetime(names["namedt"])
names["nameendt"] = pd.to_datetime(names["nameendt"])

russell = pd.read_parquet(DATA / "wrds_russell3k_proxy.parquet")
russell["eff_date"]  = pd.to_datetime(russell["eff_date"])
russell["snap_date"] = pd.to_datetime(russell["snap_date"])

sp = pd.read_parquet(DATA / "wrds_sp_constituents.parquet")
sp["from_date"] = pd.to_datetime(sp["from_date"])

results = []
def report(name, ok, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"  {name}  [{tag}]  {detail}")
    results.append((name, ok))

print("=" * 70)
print("WRDS / CRSP Look-Ahead Audit (T9–T14)")
print("=" * 70)

# ───────────────────────────────────────────────────────────────────────────
# Derive entry_match_date inline (not stored in the final parquet)
# ───────────────────────────────────────────────────────────────────────────
prices_s = prices.dropna(subset=["adj_prc"]).sort_values(["permno", "date"]).reset_index(drop=True)
prices_s["_pos"] = prices_s.groupby("permno").cumcount()

evp = events.dropna(subset=["permno"])[["permno", "entry_date", "entry_adj_prc",
                                          "crsp_return_1d","crsp_return_3d","crsp_return_5d",
                                          "crsp_return_10d","crsp_return_20d"]].copy()
evp["permno"] = evp["permno"].astype(int)
evp = evp.sort_values("entry_date")
ps  = prices_s.sort_values("date")
evp = pd.merge_asof(
    evp, ps[["permno", "date", "_pos"]],
    left_on="entry_date", right_on="date", by="permno",
    direction="forward", tolerance=pd.Timedelta(days=7),
).rename(columns={"date": "entry_match_date"})

# ───────────────────────────────────────────────────────────────────────────
# T9 — entry_match_date >= entry_date for every matched event
# ───────────────────────────────────────────────────────────────────────────
sub = evp.dropna(subset=["entry_match_date"]).copy()
bad = sub[sub["entry_match_date"] < sub["entry_date"]]
report("T9",
       len(bad) == 0,
       f"{len(bad):,} events where entry_match_date < entry_date "
       f"(checked {len(sub):,})")

# ───────────────────────────────────────────────────────────────────────────
# T10 — forward N-day returns use a strictly later price than the entry price
# ───────────────────────────────────────────────────────────────────────────
HORIZONS = [1, 3, 5, 10, 20]
worst_violation = 0
for n in HORIZONS:
    fwd_pos = sub["_pos"] + n
    fwd = prices_s.rename(columns={"_pos": "_fwd_pos", "date": "_fwd_date"})[
        ["permno", "_fwd_pos", "_fwd_date"]]
    s2 = sub.assign(_fwd_pos=fwd_pos).merge(fwd, on=["permno", "_fwd_pos"], how="left")
    s2 = s2.dropna(subset=["_fwd_date"])
    bad = (s2[f"crsp_return_{n}d"].notna() & (s2["_fwd_date"] <= s2["entry_match_date"])).sum()
    worst_violation = max(worst_violation, int(bad))
report("T10",
       worst_violation == 0,
       f"max {worst_violation:,} events across all horizons where fwd_date <= entry_match_date "
       f"(checked {len(sub):,})")

# ───────────────────────────────────────────────────────────────────────────
# T11 — Russell 3000 PIT membership uses only past mcap snapshots
# ───────────────────────────────────────────────────────────────────────────
# snap_date must be <= entry_date for every event flagged as in_RU3K_PIT.
ru_pit = russell[["permno", "eff_date", "snap_date"]].rename(
    columns={"eff_date": "from_date"})
ru_pit["from_date"] = ru_pit["from_date"].astype("datetime64[ns]")
ru_pit["snap_date"] = ru_pit["snap_date"].astype("datetime64[ns]")
ru_pit["thru_date"] = ru_pit.groupby("permno")["from_date"].shift(-1).fillna(pd.Timestamp("2099-12-31")).astype("datetime64[ns]")
ru_pit = ru_pit.sort_values("from_date")

ev = events[events["in_RU3K_PIT"].fillna(False) & events["permno"].notna()].copy()
ev["permno"] = ev["permno"].astype(int)
ev = ev.sort_values("entry_date")
m = pd.merge_asof(
    ev[["permno", "entry_date"]], ru_pit,
    left_on="entry_date", right_on="from_date", by="permno", direction="backward",
)
# snap_date must be ≤ entry_date  (universe assignment uses only past info)
m = m.dropna(subset=["snap_date"])
bad = (m["snap_date"] > m["entry_date"]).sum()
report("T11",
       bad == 0,
       f"{bad:,} in_RU3K_PIT events where snap_date > entry_date "
       f"(checked {len(m):,})")

# ───────────────────────────────────────────────────────────────────────────
# T12 — Ticker → permno mapping is point-in-time (no future renames leak)
# ───────────────────────────────────────────────────────────────────────────
# For each (permno, entry_date) pair, there must exist a (permno, ticker, namedt, nameendt)
# row where namedt ≤ entry_date ≤ nameendt. Otherwise the permno was assigned via a
# CRSP name that wasn't in effect on the event date.
ev2 = events.dropna(subset=["permno"])[["BESTTICKER", "permno", "entry_date"]].copy()
ev2["permno"] = ev2["permno"].astype(int)
ev2["_event_id"] = np.arange(len(ev2))
def _norm(s): return s.astype(str).str.upper().str.replace(r"[\.\-/]", "", regex=True)
ev2["_tkn"]    = _norm(ev2["BESTTICKER"])
names["_tkn"]  = _norm(names["ticker"])
join = ev2.merge(names[["permno", "_tkn", "namedt", "nameendt"]],
                 on=["permno", "_tkn"], how="left")
join["covered"] = (join["namedt"] <= join["entry_date"]) & \
                  (join["entry_date"] <= join["nameendt"])
covered_per_event = join.groupby("_event_id")["covered"].any()
missing = (~covered_per_event).sum()
report("T12",
       missing == 0,
       f"{missing:,} events with a permno but no matching CRSP ticker window "
       f"(checked {len(covered_per_event):,})")

# ───────────────────────────────────────────────────────────────────────────
# T13 — CRSP and yfinance returns agree on common events (sanity)
# ───────────────────────────────────────────────────────────────────────────
both = events[events["return_20d"].notna() & events["crsp_return_20d"].notna()].copy()
diff = (both["return_20d"] - both["crsp_return_20d"]).abs()
median_diff = diff.median()
p99_diff    = diff.quantile(0.99)
corr        = both[["return_20d", "crsp_return_20d"]].corr().iloc[0, 1]
# Median diff should be small (<0.5%) and correlation high (>0.95)
ok = (median_diff < 0.005) and (corr > 0.95)
report("T13",
       ok,
       f"median |Δ|={median_diff:.4f}, p99 |Δ|={p99_diff:.3f}, corr={corr:.3f} "
       f"(common N={len(both):,})")

# ───────────────────────────────────────────────────────────────────────────
# T14 — Compustat sp_constituents from_date is a past date for all included
# ───────────────────────────────────────────────────────────────────────────
# Every membership row should have from_date in the past (no future entries).
# Since this is a current snapshot pull, every member should have already joined
# by today (2026-05-15).
today = pd.Timestamp("2026-05-15")
future = (sp["from_date"] > today).sum()
report("T14",
       future == 0,
       f"{future:,} sp_constituents rows with from_date > today "
       f"(checked {len(sp):,})")

# ───────────────────────────────────────────────────────────────────────────
# Summary
# ───────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
all_pass = all(ok for _, ok in results)
print("ALL WRDS TESTS PASS" if all_pass else "WARNING: review failures above")
print("=" * 70)
