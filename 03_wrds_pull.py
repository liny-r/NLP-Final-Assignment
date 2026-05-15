"""
03 — WRDS Pull: Point-in-Time Constituents + Survivorship-Free Prices

Replaces the Wikipedia-scraped current-composition universes and yfinance
prices with WRDS data:

  - S&P 500 / 400 / 600 constituents from Compustat comp.idxcst_his
    (with from/thru date ranges — true point-in-time membership)
  - Russell 3000 proxy: CRSP top 3000 by market cap, reconstituted each
    June (matches Russell's actual annual reconstitution methodology)
  - CRSP daily prices (crsp.dsf) for all relevant PERMNOs 2010–2026,
    including delisted names — eliminates yfinance delisted-name gaps

Outputs (to data/):
  - wrds_sp_constituents.parquet   (gvkey, ticker, index, from, thru)
  - wrds_link.parquet              (gvkey -> permno via ccmxpf_lnkhist)
  - wrds_russell3k_proxy.parquet   (permno, eff_date) — annual June snapshots
  - wrds_prices.parquet            (permno, date, prc, ret, adj_prc, ...)
  - wrds_pit_membership.parquet    (ticker, date, in_SP500, in_SP1500, in_RU3K)

Run interactively the first time so the wrds package can prompt for
credentials and cache them in ~/.pgpass. Subsequent runs are non-interactive.

Usage:
    cd "/Users/yueqilin/Desktop/9796 NLP/Final Assignment"
    /Users/yueqilin/anaconda3/bin/python 03_wrds_pull.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import wrds, time, sys, os

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DATE_FROM = "2009-01-01"   # buffer for 2010 returns
DATE_TO   = "2026-06-30"

# Read username from env var WRDS_USERNAME or default (set on first interactive run)
WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "linyr")

# ───────────────────────────────────────────────────────────────────────────
# 1. Connect
# ───────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Connecting to WRDS...")
print("(First run prompts for username and password; subsequent runs use ~/.pgpass)")
print("=" * 70)
db = wrds.Connection(wrds_username=WRDS_USERNAME)
print(f"Connected as {WRDS_USERNAME}.")

# ───────────────────────────────────────────────────────────────────────────
# 2. S&P 500 / 400 / 600 historical constituents
# ───────────────────────────────────────────────────────────────────────────
# NOTE: This Compustat subscription tier returns only CURRENT S&P constituents
# (no thru_date populated). For the S&P universes we still rely on Wikipedia
# current composition; that limitation remains. The Compustat pull is kept
# because it gives us correct from_date for each current member and provides
# the gvkey→permno linkage needed for CRSP price fetches.
SP_INDEXES = {
    "SP500":  "000003",   # S&P 500 Comp-Ltd
    "SP400":  "024248",   # S&P Midcap 400 Index
    "SP600":  "030824",   # S&P Smallcap 600 Index
}

SP_CST_OUT = DATA_DIR / "wrds_sp_constituents.parquet"
if SP_CST_OUT.exists():
    sp_cst = pd.read_parquet(SP_CST_OUT)
    print(f"\n[2] Loaded cached S&P constituents: {len(sp_cst):,} rows")
else:
    print("\n[2] Pulling S&P current constituents (with join dates) from comp.idxcst_his ...")
    sp_frames = []
    for name, gvkeyx in SP_INDEXES.items():
        sql = f"""
            SELECT  i.gvkey,
                    i.gvkeyx,
                    i.from   AS from_date,
                    i.thru   AS thru_date,
                    '{name}' AS index_name
            FROM    comp.idxcst_his i
            WHERE   i.gvkeyx = '{gvkeyx}'
        """
        df = db.raw_sql(sql, date_cols=["from_date", "thru_date"])
        print(f"  {name}: {len(df):,} membership rows, {df['gvkey'].nunique():,} unique gvkeys")
        sp_frames.append(df)
    sp_cst = pd.concat(sp_frames, ignore_index=True)

    # Attach ticker from Compustat security master (no conm in this table)
    gvkeys = sp_cst["gvkey"].unique()
    print(f"  Fetching ticker for {len(gvkeys):,} unique gvkeys ...")
    sec_sql = f"""
        SELECT  gvkey, tic AS ticker
        FROM    comp.security
        WHERE   gvkey IN ({','.join(repr(g) for g in gvkeys)})
    """
    sec = db.raw_sql(sec_sql)
    sp_cst = sp_cst.merge(sec, on="gvkey", how="left")
    sp_cst.to_parquet(SP_CST_OUT)
    print(f"  Saved: {SP_CST_OUT} ({len(sp_cst):,} rows)")

# ───────────────────────────────────────────────────────────────────────────
# 3. CRSP-Compustat link (gvkey -> permno)
# ───────────────────────────────────────────────────────────────────────────
LINK_OUT = DATA_DIR / "wrds_link.parquet"
if LINK_OUT.exists():
    link = pd.read_parquet(LINK_OUT)
    print(f"\n[3] Loaded cached link table: {len(link):,} rows")
else:
    print("\n[3] Pulling CRSP-Compustat link from crsp.ccmxpf_lnkhist ...")
    gvkeys = sp_cst["gvkey"].unique()
    link_sql = f"""
        SELECT  l.gvkey,
                l.lpermno    AS permno,
                l.linkdt,
                l.linkenddt,
                l.linktype,
                l.linkprim
        FROM    crsp.ccmxpf_lnkhist l
        WHERE   l.gvkey IN ({','.join(repr(g) for g in gvkeys)})
          AND   l.linktype IN ('LC', 'LU')
          AND   l.linkprim IN ('P', 'C')
    """
    link = db.raw_sql(link_sql, date_cols=["linkdt", "linkenddt"])
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    print(f"  {len(link):,} link rows, {link['permno'].nunique():,} unique permnos")
    link.to_parquet(LINK_OUT)
    print(f"  Saved: {LINK_OUT}")

# ───────────────────────────────────────────────────────────────────────────
# 4. Russell 3000 proxy: top 3000 by market cap, reconstituted each June
# ───────────────────────────────────────────────────────────────────────────
RUSSELL_OUT = DATA_DIR / "wrds_russell3k_proxy.parquet"
if RUSSELL_OUT.exists():
    russell3k = pd.read_parquet(RUSSELL_OUT)
    print(f"\n[4] Loaded cached Russell 3000 proxy: {len(russell3k):,} rows")
else:
    print("\n[4] Building Russell 3000 proxy from CRSP monthly market cap ...")
    mcap_sql = f"""
        SELECT  m.permno,
                m.date,
                ABS(m.prc) * m.shrout * 1000.0 AS mcap_dollars
        FROM    crsp.msf m
        JOIN    crsp.msenames n
          ON    m.permno = n.permno
          AND   m.date BETWEEN n.namedt AND n.nameendt
        WHERE   m.date BETWEEN '{DATE_FROM}' AND '{DATE_TO}'
          AND   n.shrcd IN (10, 11)
          AND   n.exchcd IN (1, 2, 3)
          AND   m.prc IS NOT NULL
          AND   m.shrout IS NOT NULL
    """
    print("  Pulling monthly market cap (this may take ~2–5 minutes)...")
    t0 = time.time()
    mcap = db.raw_sql(mcap_sql, date_cols=["date"])
    print(f"  Got {len(mcap):,} monthly observations in {time.time()-t0:.1f}s")

    russell_snapshots = []
    for year in range(2010, 2027):
        target = pd.Timestamp(f"{year}-06-30")
        monthly = mcap[mcap["date"] <= target]
        if monthly.empty:
            continue
        last_date = monthly["date"].max()
        snap = monthly[monthly["date"] == last_date].nlargest(3000, "mcap_dollars")
        snap = snap.assign(eff_date=target, snap_date=last_date)
        russell_snapshots.append(snap[["permno", "eff_date", "snap_date", "mcap_dollars"]])
    russell3k = pd.concat(russell_snapshots, ignore_index=True)
    russell3k["permno"] = russell3k["permno"].astype(int)
    russell3k.to_parquet(RUSSELL_OUT)
    print(f"  Saved: {RUSSELL_OUT} ({len(russell3k):,} permno-year rows)")

# ───────────────────────────────────────────────────────────────────────────
# 5. CRSP daily prices for all permnos that ever appeared in any universe
# ───────────────────────────────────────────────────────────────────────────
PRICES_OUT = DATA_DIR / "wrds_prices.parquet"
PRICES_CHUNK_DIR = DATA_DIR / "wrds_prices_chunks"
PRICES_CHUNK_DIR.mkdir(exist_ok=True)

if PRICES_OUT.exists():
    print(f"\n[5] Cached prices file exists: {PRICES_OUT} "
          f"({PRICES_OUT.stat().st_size / 1024**2:.0f} MB) — skipping")
else:
    sp_permnos = set(link["permno"].dropna().astype(int).tolist())
    ru_permnos = set(russell3k["permno"].astype(int).tolist())
    all_permnos = sorted(sp_permnos | ru_permnos)
    print(f"\n[5] Pulling CRSP daily prices for {len(all_permnos):,} permnos ...")
    print("    (Largest pull — expect 10–30 minutes and 200–500 MB)")
    print("    Chunks are saved individually so a crash can resume.")

    CHUNK = 500
    n_chunks = (len(all_permnos) + CHUNK - 1) // CHUNK
    t0 = time.time()
    for i in range(0, len(all_permnos), CHUNK):
        chunk_id = i // CHUNK + 1
        chunk_path = PRICES_CHUNK_DIR / f"chunk_{chunk_id:04d}.parquet"
        if chunk_path.exists():
            print(f"  chunk {chunk_id}/{n_chunks}: cached, skip")
            continue
        sub = all_permnos[i:i + CHUNK]
        sql = f"""
            SELECT  permno, date, prc, ret, vol, shrout, cfacpr, cfacshr,
                    openprc, askhi, bidlo
            FROM    crsp.dsf
            WHERE   permno IN ({','.join(str(p) for p in sub)})
              AND   date BETWEEN '{DATE_FROM}' AND '{DATE_TO}'
        """
        df = db.raw_sql(sql, date_cols=["date"])
        df.to_parquet(chunk_path)
        elapsed = time.time() - t0
        print(f"  chunk {chunk_id}/{n_chunks}: {len(df):>8,} rows  ({elapsed:5.1f}s elapsed)")

    # Concatenate all chunks
    print("  Concatenating chunks ...")
    chunks = [pd.read_parquet(p) for p in sorted(PRICES_CHUNK_DIR.glob("chunk_*.parquet"))]
    prices = pd.concat(chunks, ignore_index=True)
    prices["permno"] = prices["permno"].astype(int)
    prices["adj_prc"] = prices["prc"].abs() / prices["cfacpr"].replace(0, np.nan)
    prices.to_parquet(PRICES_OUT)
    print(f"  Saved: {PRICES_OUT} "
          f"({len(prices):,} rows, "
          f"{PRICES_OUT.stat().st_size / 1024**2:.0f} MB)")

# ───────────────────────────────────────────────────────────────────────────
# 6. Build point-in-time universe-membership table
# ───────────────────────────────────────────────────────────────────────────
print("\n[6] Building PIT membership table ...")

sp_permnos = set(link["permno"].dropna().astype(int).tolist())
ru_permnos = set(russell3k["permno"].astype(int).tolist())
all_permnos = sorted(sp_permnos | ru_permnos)

# 6a. Map permno -> ticker history (from CRSP names)
NAMES_OUT = DATA_DIR / "wrds_names.parquet"
if NAMES_OUT.exists():
    names = pd.read_parquet(NAMES_OUT)
    print(f"  Loaded cached names: {len(names):,} rows")
else:
    names_sql = f"""
        SELECT  permno, namedt, nameendt, ticker, comnam
        FROM    crsp.msenames
        WHERE   permno IN ({','.join(str(p) for p in all_permnos)})
    """
    names = db.raw_sql(names_sql, date_cols=["namedt", "nameendt"])
    names["nameendt"] = names["nameendt"].fillna(pd.Timestamp("2099-12-31"))
    names = names.dropna(subset=["ticker"])
    names.to_parquet(NAMES_OUT)
    print(f"  Saved: {NAMES_OUT} ({len(names):,} rows)")

# 6b. Expand SP constituents (gvkey × date-range) → (permno × date-range) via link
sp_link = sp_cst.merge(link, on="gvkey", how="inner")
# Intersect constituent window and link window
sp_link["from_eff"] = sp_link[["from_date", "linkdt"]].max(axis=1)
sp_link["thru_eff"] = sp_link[["thru_date", "linkenddt"]].min(axis=1)
sp_link["thru_eff"] = sp_link["thru_eff"].fillna(pd.Timestamp("2099-12-31"))
sp_link = sp_link[sp_link["from_eff"] <= sp_link["thru_eff"]]
sp_link["permno"] = sp_link["permno"].astype(int)

sp_link[["permno", "index_name", "from_eff", "thru_eff", "ticker"]].to_parquet(
    DATA_DIR / "wrds_sp_permno_ranges.parquet"
)
print(f"  Saved: data/wrds_sp_permno_ranges.parquet "
      f"({len(sp_link):,} (permno, index) windows)")

# 6c. Russell 3000 PIT membership: a permno is in RU3K from year-N June 30
#     until year-(N+1) June 30 if it was in the year-N snapshot.
ru_ranges = russell3k[["permno", "eff_date"]].copy()
ru_ranges["thru_date"] = ru_ranges.groupby("permno")["eff_date"].shift(-1)
ru_ranges["thru_date"] = ru_ranges["thru_date"].fillna(pd.Timestamp("2099-12-31"))
ru_ranges["index_name"] = "RU3K"
ru_ranges = ru_ranges.rename(columns={"eff_date": "from_eff", "thru_date": "thru_eff"})

# Stash combined PIT membership ranges
pit_ranges = pd.concat([
    sp_link[["permno", "index_name", "from_eff", "thru_eff"]],
    ru_ranges[["permno", "index_name", "from_eff", "thru_eff"]],
], ignore_index=True)
pit_ranges.to_parquet(DATA_DIR / "wrds_pit_membership_ranges.parquet")
print(f"  Saved: data/wrds_pit_membership_ranges.parquet "
      f"({len(pit_ranges):,} (permno, index, from, thru) windows)")

db.close()
print("\n" + "=" * 70)
print("WRDS pull complete.")
print("=" * 70)
print("Next steps:")
print("  - 04_wrds_integrate.ipynb to join PIT membership into events_with_returns.parquet")
print("  - Re-run 01_analysis.ipynb with PIT universe flags + CRSP prices")
