"""
04 — WRDS Integration: Join CRSP prices and PIT Russell 3000 membership
into the events dataset, producing events_with_returns_wrds.parquet.

What this does (vs. the original 00_data_prep.ipynb pipeline):

  1. Maps each event's BESTTICKER to a CRSP PERMNO using msenames, with
     point-in-time correctness (ticker in effect on the event date).
  2. Recomputes 1/3/5/10/20-day forward returns from CRSP adj_prc instead
     of yfinance — eliminates yfinance's 49% RU3K coverage gap and gives
     survivorship-free prices for delisted names.
  3. Replaces in_RU3K with a survivorship-free, point-in-time flag based
     on the CRSP top-3000 market-cap proxy (annual June reconstitution).
  4. Keeps in_SP500 / in_SP1500 as-is (the Compustat subscription tier
     provides only current S&P constituents — no historical removed
     members — so we cannot eliminate the S&P survivorship bias).

Inputs (all in data/):
  - events_with_returns.parquet   (existing pipeline output)
  - wrds_prices.parquet           (from 03_wrds_pull.py)
  - wrds_names.parquet            (from 03_wrds_pull.py — ticker history)
  - wrds_russell3k_proxy.parquet  (from 03_wrds_pull.py)

Output:
  - data/events_with_returns_wrds.parquet

Run:
    /Users/yueqilin/anaconda3/bin/python 04_wrds_integrate.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("data")
HORIZONS = [1, 3, 5, 10, 20]

# ───────────────────────────────────────────────────────────────────────────
# 1. Load events + WRDS pull outputs
# ───────────────────────────────────────────────────────────────────────────
print("[1] Loading events and WRDS data ...")
events = pd.read_parquet(DATA / "events_with_returns.parquet")
events["entry_date"] = pd.to_datetime(events["entry_date"])
print(f"  events: {len(events):,} rows, {events['BESTTICKER'].nunique():,} unique tickers")

prices = pd.read_parquet(DATA / "wrds_prices.parquet",
                         columns=["permno", "date", "prc", "cfacpr", "adj_prc"])
prices["date"] = pd.to_datetime(prices["date"])
print(f"  CRSP prices: {len(prices):,} rows, {prices['permno'].nunique():,} permnos")

names = pd.read_parquet(DATA / "wrds_names.parquet")
names["namedt"]   = pd.to_datetime(names["namedt"])
names["nameendt"] = pd.to_datetime(names["nameendt"])
print(f"  CRSP names: {len(names):,} (permno, ticker, date-range) rows")

russell = pd.read_parquet(DATA / "wrds_russell3k_proxy.parquet")
russell["eff_date"] = pd.to_datetime(russell["eff_date"])
print(f"  Russell 3000 proxy: {len(russell):,} (permno, year) rows")

# ───────────────────────────────────────────────────────────────────────────
# 2. Ticker → PERMNO (PIT) mapping for each event
# ───────────────────────────────────────────────────────────────────────────
print("\n[2] Mapping BESTTICKER → PERMNO (point-in-time) ...")

# Normalise tickers to a single form (CRSP uses no separator, signal uses '.' or '-')
def _norm(s):
    return s.astype(str).str.upper().str.replace(r"[\.\-/]", "", regex=True)

events["_tkn"] = _norm(events["BESTTICKER"])
names["_tkn"]  = _norm(names["ticker"])

# Many events have the same (ticker, entry_date) — work at unique level for speed
ev_keys = events[["_tkn", "entry_date"]].drop_duplicates()
print(f"  unique (ticker, date) keys: {len(ev_keys):,}")

# merge_asof requires both frames sorted by the `on` column globally
ev_keys = ev_keys.dropna(subset=["entry_date"]).sort_values("entry_date")
names_s = names.dropna(subset=["namedt"]).sort_values("namedt")

mapped = pd.merge_asof(
    ev_keys, names_s, left_on="entry_date", right_on="namedt", by="_tkn",
    direction="backward",
)
# Discard rows where entry_date > nameendt (name not in effect anymore)
mapped = mapped[(mapped["entry_date"] >= mapped["namedt"]) &
                (mapped["entry_date"] <= mapped["nameendt"])]
mapped = mapped[["_tkn", "entry_date", "permno"]].dropna()
mapped["permno"] = mapped["permno"].astype(int)
print(f"  matched: {len(mapped):,} / {len(ev_keys):,} keys "
      f"({100*len(mapped)/len(ev_keys):.1f}%)")

events = events.merge(mapped, on=["_tkn", "entry_date"], how="left")
print(f"  events with permno: {events['permno'].notna().sum():,} / {len(events):,} "
      f"({100*events['permno'].notna().mean():.1f}%)")

# ───────────────────────────────────────────────────────────────────────────
# 3. Recompute forward returns from CRSP adj_prc
# ───────────────────────────────────────────────────────────────────────────
print("\n[3] Recomputing forward returns from CRSP adjusted prices ...")
# Build per-permno daily adj_prc, sorted, indexed by date
prices_s = prices[["permno", "date", "adj_prc"]].dropna(subset=["adj_prc"])
prices_s = prices_s.sort_values(["permno", "date"]).reset_index(drop=True)

# Entry price: the close at entry_date (or next CRSP trading day if entry is non-trading)
ev_pr = events[["permno", "entry_date"]].dropna().drop_duplicates()
ev_pr["permno"] = ev_pr["permno"].astype(int)
ev_pr = ev_pr.sort_values("entry_date")
prices_s_sorted = prices_s.sort_values("date")
entry_match = pd.merge_asof(
    ev_pr, prices_s_sorted, left_on="entry_date", right_on="date", by="permno",
    direction="forward", tolerance=pd.Timedelta(days=7),
).rename(columns={"adj_prc": "entry_adj_prc", "date": "entry_match_date"})

# Build a permno→date→adj_prc index for fast N-day forward lookup
pidx = prices_s.set_index(["permno", "date"])["adj_prc"]

# For each horizon N, find the N-th CRSP trading day on/after entry_match_date
# Use position-based lookup: for each (permno, entry_match_date), get index pos in prices_s
prices_pos = prices_s.copy()
prices_pos["_pos"] = prices_pos.groupby("permno").cumcount()

entry_match = entry_match.merge(
    prices_pos.rename(columns={"date": "entry_match_date"})[["permno", "entry_match_date", "_pos"]],
    on=["permno", "entry_match_date"], how="left",
)

forward_cols = {}
for n in HORIZONS:
    # nth forward day per permno
    fwd = prices_pos.assign(_pos_back=prices_pos["_pos"] - n)
    fwd = fwd[fwd["_pos_back"] >= 0]
    fwd = fwd[["permno", "_pos_back", "adj_prc"]].rename(
        columns={"_pos_back": "_pos", "adj_prc": f"_fwd_adj_{n}"})
    entry_match = entry_match.merge(fwd, on=["permno", "_pos"], how="left")
    entry_match[f"crsp_return_{n}d"] = (
        entry_match[f"_fwd_adj_{n}"] / entry_match["entry_adj_prc"] - 1
    )
    forward_cols[f"crsp_return_{n}d"] = None

events = events.merge(
    entry_match[["permno", "entry_date"] +
                ["entry_adj_prc"] +
                [f"crsp_return_{n}d" for n in HORIZONS]],
    on=["permno", "entry_date"], how="left",
)

# Coverage stats
for n in HORIZONS:
    yf_n  = events[f"return_{n}d"].notna().sum()
    cr_n  = events[f"crsp_return_{n}d"].notna().sum()
    both  = (events[f"return_{n}d"].notna() & events[f"crsp_return_{n}d"].notna()).sum()
    only_crsp = (events[f"return_{n}d"].isna() & events[f"crsp_return_{n}d"].notna()).sum()
    print(f"  {n:>2}d: yfinance={yf_n:>7,}  crsp={cr_n:>7,}  "
          f"both={both:>7,}  crsp-only(new)={only_crsp:>7,}")

# ───────────────────────────────────────────────────────────────────────────
# 4. PIT Russell 3000 membership
# ───────────────────────────────────────────────────────────────────────────
print("\n[4] Building PIT in_RU3K_PIT flag from CRSP top-3000 mcap ...")

# A permno is in_RU3K at date d if it was in the snapshot for the most recent
# June 30 on or before d.
russell["eff_date"]  = pd.to_datetime(russell["eff_date"]).astype("datetime64[ns]")
russell["thru_date"] = russell.groupby("permno")["eff_date"].shift(-1)
russell["thru_date"] = russell["thru_date"].fillna(pd.Timestamp("2099-12-31"))
ru_pit = russell[["permno", "eff_date", "thru_date"]].rename(
    columns={"eff_date": "from_date"})
ru_pit["from_date"] = ru_pit["from_date"].astype("datetime64[ns]")
ru_pit["thru_date"] = ru_pit["thru_date"].astype("datetime64[ns]")

# Match each event's (permno, entry_date) to a Russell window
events_ru = events[["permno", "entry_date"]].dropna().drop_duplicates()
events_ru["permno"] = events_ru["permno"].astype(int)
events_ru = events_ru.sort_values("entry_date")
ru_pit_s = ru_pit.sort_values("from_date")
in_ru = pd.merge_asof(
    events_ru, ru_pit_s, left_on="entry_date", right_on="from_date", by="permno",
    direction="backward",
)
in_ru["in_RU3K_PIT"] = (in_ru["from_date"].notna() &
                       (in_ru["entry_date"] < in_ru["thru_date"])).astype(bool)
in_ru = in_ru[["permno", "entry_date", "in_RU3K_PIT"]]

events = events.merge(in_ru, on=["permno", "entry_date"], how="left")
events["in_RU3K_PIT"] = events["in_RU3K_PIT"].fillna(False)
print(f"  in_RU3K (current, exchange-flag): {events['in_RU3K'].sum():>7,}")
print(f"  in_RU3K_PIT (CRSP mcap, PIT):     {events['in_RU3K_PIT'].sum():>7,}")
print(f"  RU3K_PIT but not RU3K:            {(events['in_RU3K_PIT'] & ~events['in_RU3K']).sum():>7,} (delisted members)")
print(f"  RU3K but not RU3K_PIT:            {(~events['in_RU3K_PIT'] & events['in_RU3K']).sum():>7,} (in exchange list, not top-3000)")

# ───────────────────────────────────────────────────────────────────────────
# 5. Save
# ───────────────────────────────────────────────────────────────────────────
out = DATA / "events_with_returns_wrds.parquet"
events.drop(columns=["_tkn"], errors="ignore").to_parquet(out)
print(f"\n[5] Saved: {out}  ({out.stat().st_size / 1024**2:.0f} MB)")
print("\nNext: re-run 01_analysis.ipynb with `events_with_returns_wrds.parquet`")
print("       and `crsp_return_{N}d` / `in_RU3K_PIT` columns for the survivorship-free run.")
