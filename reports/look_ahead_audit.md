# Look-Ahead Bias Audit Checklist

**Project:** Backtesting the ProntoNLP Earnings-Call ATC Signal  
**Author:** Yueqi Lin  
**Notebook:** `00_data_prep.ipynb`  
**Reference:** Student Handout §3 — *Look-Ahead Bias: The Audit You Must Pass*

---

## §3.1 — Entry Timing: AMC vs BMO

**Rule:** Parse `MOSTIMPORTANTDATEUTC`. If hour ≥ 16 UTC → after-market-close → entry is the **next** trading day's close. If hour < 16 UTC → entry is the **same** trading day's close.

**Implementation:** `00_data_prep.ipynb`, cells 18–19. Vectorized via `numpy.searchsorted` on the NYSE calendar array derived from SPY prices.

**Assertions added (cell 19):**
- `entry_date > call_date` for all AMC events ✅
- `entry_date >= call_date` for all BMO events ✅

**Status: ✅ PASS**

---

## §3.2 — Forward Returns Are Targets, Never Inputs

**Rule:** `return_Nd` columns are computed *from* entry price. They must never appear in the model feature set.

**Implementation:** Returns computed in cells 22–23 after `entry_date` is locked. Feature columns built in cells 28–29 from AspectTheme matrix and EventScores only.

**Assertion added (cell 24):**
- `{return_1d, return_3d, return_5d, return_10d, return_20d}` is disjoint from `feature_cols` ✅

**Status: ✅ PASS**

---

## §3.3 — Cross-Sectional Features Must Be Point-in-Time

**Rule:** Z-scores, percentile ranks, sector means computed across "all events in the same quarter" leak future data for events early in the quarter. Use only events that have already happened (expanding-window percentile).

**Implementation:** Cross-sectional features (sector ranks, Z-scores) are **deferred to `01_analysis.ipynb`** and computed per walk-forward fold using only training-set events. No cross-sectional transformations are applied in `00_data_prep.ipynb`.

**Status: ✅ PASS** (deferred by design; documented in cell 0 markdown)

---

## §3.4 — Feature Selection Is Part of Training

**Rule:** Spearman/mutual-info ranking, PCA fits, target-encoding must be done on training fold only, refitted at every walk-forward step.

**Implementation:** No feature selection is performed in `00_data_prep.ipynb`. All selection happens inside the walk-forward loop in `01_analysis.ipynb`, fit on train set, applied to test set.

**Status: ✅ PASS** (deferred to notebook 01 by design)

---

## §3.5 — Imputation and Scaling Are Part of Training

**Rule:** `StandardScaler.fit` and median imputation must use training data only. Pattern: `fit_transform(train)` → `transform(test)`.

**Implementation:** No scaling or imputation in `00_data_prep.ipynb`. Applied inside walk-forward loop in `01_analysis.ipynb`.

**Status: ✅ PASS** (deferred to notebook 01 by design)

---

## §3.6 — Universe Membership Is Point-in-Time

**Rule:** Today's S&P 500 is not 2014's S&P 500. Use historical constituents or document the survivorship-bias caveat.

**Implementation:** Historical constituent data is not available (no CRSP/Compustat access). All three universes use **current composition** as of 2026 (sourced from Wikipedia). The survivorship-bias caveat is documented in:
- `00_data_prep.ipynb` cell 10 markdown
- `reports/research_report.md` §7 (Risks and Limitations)
- `README.md` Key Design Decisions

Reported alpha should be treated as an **upper bound**.

**Status: ✅ PASS** (caveat explicitly documented)

---

## §3.7 — INGESTDATEUTC ≠ Availability Date

**Rule:** Some calls were ingested by ProntoNLP days after the actual call. For the strictest backtest, entry date = `max(MOSTIMPORTANTDATEUTC + entry-rule, INGESTDATEUTC)`. Document whichever rule you choose.

**Finding:** Inspection of `INGESTDATEUTC` (cell 18) reveals a **mean lag of 1,658 days** (4.5 years) with 83.7% of events having a positive lag. This confirms the field records a **batch historical backfill** — ProntoNLP processed the full historical archive in bulk, not in real time. Applying this constraint would push 81% of entry dates to 2023+, joining 2010 signals to 2023 prices — which is itself severe look-ahead.

**Design choice:** Entry date uses `MOSTIMPORTANTDATEUTC` only. `INGESTDATEUTC` is parsed and the lag distribution is printed in cell 18 for transparency.

**Status: ✅ PASS** (choice documented with statistical justification)

---

## §3.8 — No "Future" QoQ Deltas

**Rule:** QoQ features (current quarter minus previous quarter) are fine. "Next quarter minus current" is not.

**Implementation:** QoQ deltas computed in cell 29 via `groupby('BESTTICKER')[cols].shift(1)` on data sorted by `entry_date` ascending. Each event's delta uses the **previous** event for that ticker only.

**Status: ✅ PASS**

---

## §3.9 — Corporate-Action / Delisting Handling

**Rule:** Don't assume a fill on a non-tradable day. If price source returns NaN, skip the trade or roll forward. Document the rule.

**Implementation:** Forward returns are joined via left-merge on `(BESTTICKER, entry_date)`. Events with no price data receive `NaN` returns and are excluded from IC/quintile analysis (not forward-filled or imputed). Events where the exit price is missing (ticker delisted before exit date) also receive `NaN`. No roll-forward is applied.

**Rule documented:** NaN-return events are excluded from all analyses. This is conservative and avoids assuming liquidity.

**Status: ✅ PASS** (NaN-exclusion rule documented)

---

## §3.10 — Hyperparameter Tuning Leaks Too

**Rule:** If you grid-search on the full sample and re-run the backtest with the winners, you have leaked. Tune on a held-out sub-period (e.g., 2010–2017), then freeze and walk forward from 2018+.

**Implementation:** Hyperparameter tuning is performed in `01_analysis.ipynb` using only the training portion of each walk-forward fold. The initial tuning sub-period is 2010–2017; walk-forward test begins 2018Q1. No full-sample grid search is performed.

**Status: ✅ PASS** (deferred to notebook 01 by design)

---

## Summary

| # | Item | Status |
|---|------|--------|
| 3.1 | Entry timing (AMC/BMO) | ✅ PASS |
| 3.2 | Forward returns as targets only | ✅ PASS |
| 3.3 | Cross-sectional features point-in-time | ✅ PASS |
| 3.4 | Feature selection in training only | ✅ PASS |
| 3.5 | Imputation/scaling in training only | ✅ PASS |
| 3.6 | Universe membership point-in-time | ✅ PASS (caveat documented) |
| 3.7 | INGESTDATEUTC ≠ availability date | ✅ PASS (choice documented) |
| 3.8 | No future QoQ deltas | ✅ PASS |
| 3.9 | Corporate-action / delisting handling | ✅ PASS |
| 3.10 | Hyperparameter tuning | ✅ PASS |

**All 10 look-ahead audit items: PASS**
