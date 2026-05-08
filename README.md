# Backtesting the ProntoNLP Earnings-Call ATC Signal

**Course:** LLM-Driven Quant Research  
**Author:** Yueqi Lin  
**Dataset:** `Earnings_ATC_until_2026-04-21.csv` (~4.5 GB, not included — see §Data)

---

## Overview

A rigorous, look-ahead-free backtest of the ProntoNLP Earnings-Call ATC (Aspect-Theme Classifier) signal across three equity universes: **S&P 500**, **S&P 1500**, and **Russell 3000**. The project spans earnings calls from 2010–2026 (~376 K events) and evaluates the signal at daily, weekly, and monthly rebalancing cadences.

---

## Repository Structure

```
├── 00_data_prep.ipynb          # Data pipeline: CSV → Parquet → features → returns
├── 01_analysis.ipynb           # IC, portfolios, walk-forward models, robustness
├── 02_lookahead_tests.ipynb    # 8 programmatic look-ahead bias tests T1–T8 (all pass)
├── Makefile                    # One-command reproduce: make all
├── build_charts_pdf.py         # Bundles reports/output/*.png into PDF
├── reports/
│   ├── look_ahead_audit.md     # §3 look-ahead bias checklist (required deliverable)
│   ├── research_report.md      # Full research write-up (source for PDF)
│   ├── research_report.pdf     # Compiled report
│   ├── backtest_charts.pdf     # All 21 figures bundled
│   └── output/                 # 24 PNG figures (committed, regenerable)
├── data/
│   └── universes.json          # SP500 / SP1500 / RU3K ticker lists (committed)
│   # signals.parquet, prices.parquet, events_with_returns.parquet,
│   # signal_slices.parquet, sparse_features.parquet are gitignored (regenerable)
├── Student_Handout_Earnings_ATC_Backtest.pdf
└── README.md
```

---

## Reproducing Results

### Quick start (fresh machine, one command)

```bash
# 1. Clone the repo and cd into it
git clone <repo-url> && cd <repo-dir>

# 2. Create environment and install dependencies
conda create -n atc python=3.11 -y
conda activate atc
pip install pandas pyarrow yfinance lightgbm xgboost scikit-learn \
            tqdm requests jupyter nbconvert matplotlib scipy pandoc

# 3. Place the raw CSV in the project root
#    Earnings_ATC_until_2026-04-21.csv (~4.5 GB) — request from instructor

# 4. Run everything
make all    # data prep → analysis → look-ahead tests → PDF report → charts PDF
```

`make all` runs the full pipeline end-to-end. Individual targets:

| Target | What it does |
|--------|-------------|
| `make data` | Run `00_data_prep.ipynb` (CSV → Parquet, ~30–60 min) |
| `make analysis` | Run `01_analysis.ipynb` (IC, portfolios, models, ~20–30 min) |
| `make tests` | Run `02_lookahead_tests.ipynb` (look-ahead audit, ~2 min) |
| `make report` | Compile `reports/research_report.pdf` via pandoc |
| `make charts` | Build `reports/backtest_charts.pdf` from PNGs |
| `make clean` | Remove generated Parquet files and PNGs |

### Manual step-by-step

1. **Prerequisites:** Install dependencies above, place raw CSV in project root.
2. **Data prep:** Run `00_data_prep.ipynb` top-to-bottom (Kernel → Restart & Run All). Outputs saved to `data/`:

| File | Size | Description |
|------|------|-------------|
| `signals.parquet` | ~320 MB | Cleaned signal rows (non-delete, Fluff/Filler dropped) |
| `prices.parquet` | ~42 MB | Daily adj-close for all universe tickers |
| `events_with_returns.parquet` | ~500 MB | Total-slice events + 772 features + 5 forward returns |
| `sparse_features.parquet` | ~42 MB | 376,790 rows × 405 raw AspectTheme columns |
| `signal_slices.parquet` | ~35 MB | ATCClassifierScore + EventScores for Total/CEO/CFO/Analysts |
| `universes.json` | <1 MB | SP500/SP1500/RU3K ticker lists |

> **Subsequent runs are incremental.** `signals.parquet` and `universes.json` are skipped if already present.

3. **Analysis:** Run `01_analysis.ipynb` top-to-bottom (~20–30 min). Figures saved to `reports/output/`.
4. **Look-ahead tests:** Run `02_lookahead_tests.ipynb` (~2 min). All 8 tests pass; figures saved to `reports/output/`.
5. **PDF report:** `cd reports && pandoc research_report.md -o research_report.pdf --pdf-engine=xelatex ...` (or `make report`).

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
| Look-ahead bias audit (10-item checklist §3.1–§3.10, signed) | ✅ Complete | `reports/look_ahead_audit.md` |
| Formal look-ahead bias tests (8 programmatic tests T1–T8) | ✅ Complete | `02_lookahead_tests.ipynb` |
| Analysis notebook | ✅ Complete | `01_analysis.ipynb` |
| Research PDF (with embedded figures) | ✅ Complete | `reports/research_report.pdf` |
| Backtest charts (21 figures) | ✅ Complete | `reports/output/` + `reports/backtest_charts.pdf` |
| One-command reproducibility | ✅ Complete | `Makefile` — run `make all` |
