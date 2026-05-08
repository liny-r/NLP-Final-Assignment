# Backtesting the ProntoNLP Earnings-Call ATC Signal

**Course:** LLM-Driven Quant Research  
**Author:** Rose Lin  
**Dataset:** `Earnings_ATC_until_2026-04-21.csv` (~4.5 GB, not included — see §Data)

---

## Overview

A rigorous, look-ahead-free backtest of the ProntoNLP Earnings-Call ATC (Aspect-Theme Classifier) signal across three equity universes: **S&P 500**, **S&P 1500**, and **Russell 3000**. The project spans earnings calls from 2010–2026 (~376 K events) and evaluates the signal at daily, weekly, and monthly rebalancing cadences.

---

## Repository Structure

```
├── 00_data_prep.ipynb          # Data pipeline: CSV → Parquet → features → returns
├── reports/
│   ├── look_ahead_audit.md     # §3 look-ahead bias checklist (required deliverable)
│   └── research_report.md      # Full research write-up (source for PDF)
├── data/
│   └── universes.json          # SP500 / SP1500 / RU3K ticker lists (committed)
│   # signals.parquet, prices.parquet, events_with_returns.parquet
│   # and signal_slices.parquet are gitignored (regenerable)
├── results/                    # Output figures and tables (gitignored)
├── Student_Handout_Earnings_ATC_Backtest.pdf
└── README.md
```

---

## Reproducing Results

### 1. Prerequisites

```bash
conda create -n atc python=3.11
conda activate atc
pip install pandas pyarrow yfinance lightgbm scikit-learn tqdm requests
```

### 2. Obtain the data

Place `Earnings_ATC_until_2026-04-21.csv` in the project root (not included in repo — request from instructor).

### 3. Run the data pipeline

Open and run **`00_data_prep.ipynb`** top-to-bottom (Kernel → Restart & Run All).

- Runtime: ~30–60 min (dominated by yfinance price fetch on first run)
- Outputs written to `data/`:

| File | Size | Description |
|------|------|-------------|
| `signals.parquet` | ~320 MB | Cleaned signal rows (non-delete, Fluff/Filler dropped) |
| `prices.parquet` | ~42 MB | Daily adj-close for all universe tickers |
| `events_with_returns.parquet` | ~204 MB | Total-slice events + 344 features (86 base + QoQ/2Q/YoY trend variants, winsorized returns) + 5 forward returns |
| `sparse_features.parquet` | ~42 MB | 376,790 rows × 405 raw AspectTheme columns (Stretch model input) |
| `signal_slices.parquet` | ~35 MB | ATCClassifierScore + EventScores for Total/CEO/CFO/Analysts |
| `universes.json` | <1 MB | SP500/SP1500/RU3K ticker lists |

> **Subsequent runs are incremental.** `signals.parquet` and `universes.json` are skipped if already present. Price fetching picks up from where it left off.

### 4. Run the analysis

Open and run **`01_analysis.ipynb`** top-to-bottom. Covers IC analysis (all 3 universes, sector, feature×horizon heatmap), quintile/decile portfolios, walk-forward Ridge + LightGBM (3 tiers), cadence/turnover/exposure analysis, and parameter sensitivity. Runtime: ~20–30 min (dominated by the walk-forward Stretch model).

---

## Key Design Decisions

### Entry timing (§3.1 — no look-ahead)
- Call hour **≥ 16 UTC** (after-market close): entry at **next** NYSE trading day close
- Call hour **< 16 UTC** (before/during market): entry at **same** NYSE trading day close
- NYSE calendar sourced from SPY daily prices

### INGESTDATEUTC (§3.7 — documented)
Mean ingestion lag is **1,658 days** — confirming this field records a batch historical backfill, not real-time data availability. Entry dates are based on `MOSTIMPORTANTDATEUTC` only. This choice is documented in `00_data_prep.ipynb` cell 18 and in the look-ahead audit.

### Universe membership
All three universes use **current composition** (no point-in-time constituent data available). Reported alpha is an **upper bound**; survivorship bias caveat is stated in all results.

### Transaction costs
**5 bps one-way** assumed throughout, per handout §2.2.

---

## Deliverables

| Item | Status | Location |
|------|--------|----------|
| Data pipeline | ✅ Complete | `00_data_prep.ipynb` |
| Look-ahead audit checklist | ✅ Complete | `reports/look_ahead_audit.md` |
| Formal look-ahead bias tests (5 tests) | ✅ Complete | `02_lookahead_tests.ipynb` |
| Analysis notebook | ✅ Complete | `01_analysis.ipynb` |
| Research PDF (with embedded figures) | ✅ Complete | `reports/research_report.pdf` |
| Backtest charts (15 figures) | ✅ Complete | `results/` |
| One-command reproducibility | ✅ Complete | `Makefile` — run `make all` |
