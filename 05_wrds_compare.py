"""
05 — WRDS vs. yfinance head-to-head comparison

Recomputes the headline metrics (Spearman IC, monthly quintile L/S Sharpe,
walk-forward Ridge/LightGBM Sharpe) on three configurations:

  (A) yfinance prices + Wikipedia universes  (the published results)
  (B) CRSP prices       + same Wikipedia universes (price-quality only)
  (C) CRSP prices       + CRSP PIT Russell 3000 (survivorship-free)

Output: a Markdown table comparing the three runs.

Run:
    /Users/yueqilin/anaconda3/bin/python 05_wrds_compare.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
TC_BPS = 5
HORIZON_RET = "return_20d"     # for yfinance
HORIZON_CRSP = "crsp_return_20d"

# ───────────────────────────────────────────────────────────────────────────
print("Loading events_with_returns_wrds.parquet ...")
df = pd.read_parquet(DATA / "events_with_returns_wrds.parquet")
df["entry_date"] = pd.to_datetime(df["entry_date"])
print(f"  {len(df):,} events, {df['BESTTICKER'].nunique():,} unique tickers")
print()

# ───────────────────────────────────────────────────────────────────────────
# 1. Spearman IC: ATCClassifierScore → return_20d, by universe
# ───────────────────────────────────────────────────────────────────────────
def spearman_ic(signal, ret):
    m = signal.notna() & ret.notna()
    if m.sum() < 50: return np.nan, m.sum()
    return spearmanr(signal[m], ret[m])[0], m.sum()

print("=" * 72)
print("(1) Spearman IC of ATCClassifierScore vs. 20d return")
print("=" * 72)
configs = [
    ("(A) yfinance + Wiki SP500",  df["in_SP500"].fillna(False),   df[HORIZON_RET]),
    ("(A) yfinance + Wiki SP1500", df["in_SP1500"].fillna(False),  df[HORIZON_RET]),
    ("(A) yfinance + Wiki RU3K",   df["in_RU3K"].fillna(False),    df[HORIZON_RET]),
    ("(B) CRSP     + Wiki SP500",  df["in_SP500"].fillna(False),   df[HORIZON_CRSP]),
    ("(B) CRSP     + Wiki SP1500", df["in_SP1500"].fillna(False),  df[HORIZON_CRSP]),
    ("(B) CRSP     + Wiki RU3K",   df["in_RU3K"].fillna(False),    df[HORIZON_CRSP]),
    ("(C) CRSP     + PIT  RU3K",   df["in_RU3K_PIT"].fillna(False),df[HORIZON_CRSP]),
]
print(f"{'Config':40s}  {'N':>8}  {'IC_20d':>9}")
for name, mask, ret in configs:
    sub = df[mask]
    ic, n = spearman_ic(sub["ATCClassifierScore"], sub[ret.name])
    print(f"  {name:40s}  {n:>8,}  {ic:+.4f}")
print()

# ───────────────────────────────────────────────────────────────────────────
# 2. Monthly Quintile L/S Sharpe by universe
# ───────────────────────────────────────────────────────────────────────────
def quintile_sharpe(sub_df, score_col, ret_col, tc_bps=TC_BPS):
    s = sub_df[sub_df[score_col].notna() & sub_df[ret_col].notna()].copy()
    s["_period"] = s["entry_date"].dt.to_period("M")
    recs = []
    for period, g in s.groupby("_period"):
        if len(g) < 10: continue
        try:
            g = g.copy()
            g["_q"] = pd.qcut(g[score_col], 5, labels=False, duplicates="drop")
            if g["_q"].nunique() < 5: continue
            qm = g.groupby("_q")[ret_col].mean()
            ls = qm.iloc[-1] - qm.iloc[0]
            recs.append({"period": period.to_timestamp(),
                         "LS_net": ls - 4 * tc_bps / 10_000})
        except Exception:
            continue
    if not recs: return np.nan, np.nan, 0
    res = pd.DataFrame(recs).set_index("period").sort_index()
    sh  = res["LS_net"].mean() / res["LS_net"].std() * np.sqrt(12) if res["LS_net"].std() > 0 else np.nan
    cum = (1 + res["LS_net"]).cumprod()
    dd  = float((cum / cum.cummax() - 1).min())
    return sh, dd, len(res)

print("=" * 72)
print("(2) Monthly quintile L/S net Sharpe (20d return, 5 bps one-way TC)")
print("=" * 72)
port_cfgs = [
    ("(A) yfinance + Wiki SP500",  "in_SP500",     HORIZON_RET),
    ("(A) yfinance + Wiki SP1500", "in_SP1500",    HORIZON_RET),
    ("(A) yfinance + Wiki RU3K",   "in_RU3K",      HORIZON_RET),
    ("(B) CRSP     + Wiki SP500",  "in_SP500",     HORIZON_CRSP),
    ("(B) CRSP     + Wiki SP1500", "in_SP1500",    HORIZON_CRSP),
    ("(B) CRSP     + Wiki RU3K",   "in_RU3K",      HORIZON_CRSP),
    ("(C) CRSP     + PIT  RU3K",   "in_RU3K_PIT",  HORIZON_CRSP),
]
print(f"{'Config':40s}  {'Sharpe_net':>10}  {'Max DD':>8}  {'N':>4}")
for name, mask_col, ret_col in port_cfgs:
    sub = df[df[mask_col].fillna(False)]
    sh, dd, n = quintile_sharpe(sub, "ATCClassifierScore", ret_col)
    print(f"  {name:40s}  {sh:>+10.2f}  {dd*100:>+7.1f}%  {n:>4}")
print()

# ───────────────────────────────────────────────────────────────────────────
# 3. Walk-forward Ridge on 772 features (sanity check)
# ───────────────────────────────────────────────────────────────────────────
META = {"DocDate","BESTTICKER","SECTOR","QTR_YEAR","entry_date","year",
        "in_SP500","in_SP1500","in_RU3K","in_RU3K_PIT","permno",
        "entry_adj_prc","entry_match_date","_pos"}
RET  = {"return_1d","return_3d","return_5d","return_10d","return_20d",
        "crsp_return_1d","crsp_return_3d","crsp_return_5d","crsp_return_10d","crsp_return_20d"}
FEAT = [c for c in df.columns if c not in META | RET]
quarters = pd.period_range("2018Q1", "2026Q2", freq="Q")
print(f"Walk-forward feature set: {len(FEAT)} features")

def wf_ridge(use_ret):
    ics = []
    for q in quarters:
        train = df[(df["entry_date"] < q.start_time) & df[use_ret].notna()]
        test  = df[(df["entry_date"] >= q.start_time) & (df["entry_date"] <= q.end_time) & df[use_ret].notna()]
        if len(train) < 200 or len(test) < 20: continue
        scaler = StandardScaler().fit(train[FEAT].fillna(0).values.astype("float32"))
        Xtr = scaler.transform(train[FEAT].fillna(0).values.astype("float32"))
        Xte = scaler.transform(test[FEAT].fillna(0).values.astype("float32"))
        ridge = RidgeCV(alphas=[0.01,0.1,1,10,100,1000], fit_intercept=True, scoring="neg_mean_squared_error")
        ridge.fit(Xtr, train[use_ret].values)
        p = ridge.predict(Xte)
        m = ~np.isnan(p) & test[use_ret].notna().values
        if m.sum() < 20: continue
        ic = spearmanr(p[m], test[use_ret].values[m])[0]
        ics.append(ic)
    return np.array(ics)

print("=" * 72)
print("(3) Walk-Forward Ridge (all-universe, 34 quarters): IC IR")
print("=" * 72)
for name, ret_col in [("yfinance return_20d", HORIZON_RET),
                     ("CRSP crsp_return_20d", HORIZON_CRSP)]:
    ics = wf_ridge(ret_col)
    if len(ics) == 0:
        print(f"  {name:30s}  insufficient data")
        continue
    ir = ics.mean() / ics.std() if ics.std() > 0 else np.nan
    print(f"  {name:30s}  mean IC = {ics.mean():+.4f}  IR = {ir:+.2f}  n = {len(ics)}")
print()
print("Done.")
