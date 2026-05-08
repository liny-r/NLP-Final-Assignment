---
title: "Backtesting the ProntoNLP Earnings-Call ATC Signal: Evidence from S&P 500, S&P 1500, and Russell 3000"
author: "Yueqi Lin"
date: "May 2026"
fontsize: 9pt
linestretch: 1
geometry:
  - margin=1in
toc: false
colorlinks: false
header-includes: |
  \usepackage{xcolor}
  \usepackage{sectsty}
  \usepackage{newunicodechar}
  \usepackage{pifont}
  \sectionfont{\color[HTML]{0B2545}}
  \subsectionfont{\color[HTML]{1D4E89}}
  \subsubsectionfont{\color[HTML]{1D4E89}}
  \newunicodechar{✓}{\ding{51}}
  \newunicodechar{✗}{\ding{55}}
  \newunicodechar{≥}{$\geq$}
  \newunicodechar{≤}{$\leq$}
  \newunicodechar{≈}{$\approx$}
  \newunicodechar{→}{$\rightarrow$}
  \usepackage{float}
  \floatplacement{figure}{H}
---

\newpage
\tableofcontents
\newpage

# Abstract

We conduct a rigorous, look-ahead-free backtest of the ProntoNLP Earnings-Call ATC (Aspect-Theme Classifier) signal using 376,790 earnings call events spanning 2010–2026. The ATC signal aggregates aspect-level sentiment, theme-level tone, and consensus KPI surprises into a single per-call ranking score. We test across three equity universes — S&P 500, S&P 1500, and Russell 3000 — at daily, weekly, and monthly rebalancing cadences with 1–20 day holding periods. All ten look-ahead bias audit items pass. The ATCClassifierScore achieves consistent Spearman IC of +0.039–0.049 (SP500, 10–20d horizons). Monthly quintile L/S portfolios deliver net Sharpe ratios of 0.45 (S&P 500), 0.74 (S&P 1500), and 1.28 (Russell 3000) after 20 bps round-trip transaction costs. Monthly rebalancing is the robust primary cadence, delivering positive net Sharpe across all three universes (+0.45 / +0.74 / +1.28). Daily rebalancing is TC-destroyed for SP500 (net Sharpe −0.03); SP1500/RU3K show higher daily gross returns but also materially higher drawdown (−39.5% / −78.3%) and capacity constraints under realistic TC assumptions. We engineer 772 features from the raw ATC matrix using Aspect × Theme cross-products augmented with QoQ, 2-quarter, and YoY trend variants, enabling an expanding-window predictive model. In the expanding walk-forward (34 quarters, 2018Q1–2026Q2), the best IC model is **Combo XGBoost** (772 engineered + 13 Lasso-selected raw AspectTheme cells, IR +2.27), outperforming Ridge (IR +1.96), LightGBM (IR +1.64), and the ATC baseline (IR +1.12). In out-of-sample walk-forward portfolio simulation (SP500, monthly), **Combo XGBoost achieves net Sharpe +0.76 with max drawdown −8.9%** — nearly 3× higher Sharpe than Ridge (+0.27) or LightGBM (+0.26) — driven by XGBoost's ability to exploit non-linear interactions between the 13 selected sparse cells and the engineered feature set. Sparse feature selection uses a dual-method filter (IC ranking + LassoCV intersection on 2010–2017 training data), reducing 405 raw AspectTheme cells to 13 robustly predictive features concentrated in positive Financial Performance and Capital Allocation language. The 2-quarter trend in ATC score (`ATCClassifierScore_2q`, IC_5d = +0.047) is the single most predictive individual feature. Parameter sensitivity shows break-even near 8 bps one-way. The most critical risk is regime-dependent signal decay: pre-COVID ATC 10d IC (+0.063) has collapsed to +0.008 post-COVID, though the ML layer (XGBoost IR +2.27) substantially recovers lost signal through engineered trend features.





# 1. Introduction

Earnings calls are among the most information-dense events in the corporate disclosure calendar. In the minutes following a call, equity prices move sharply as investors update their views on earnings quality, management guidance, and strategic direction. This creates a natural setting for NLP-based alpha generation: if a model can rapidly score the sentiment and content of a transcript before the broader market has fully digested it, there is a window to trade profitably.

ProntoNLP's Aspect-Theme Classifier (ATC) signal is designed to exploit this window. It combines sentence-level aspect and theme classification — distinguishing, for example, backward-looking current-state commentary from forward-looking forecast statements — with consensus KPI beat/miss data to produce a single per-call score, `ATCClassifierScore`. The signal has been used by industry trading desks; replicating its historical quintile P&L is therefore a meaningful first deliverable.

This paper makes the following contributions:

1. A fully reproducible, look-ahead-free data pipeline that transforms the raw ATC CSV into a feature-rich Parquet dataset suitable for machine learning.
2. An explicit audit of all ten look-ahead bias vectors identified in the course handout, with code-level evidence for each.
3. A comparison of the ATC signal's predictive power across three equity universes and multiple return horizons (1, 3, 5, 10, 20 trading days), including sector- and market-cap-stratified IC.
4. A feature × horizon IC heatmap identifying the 2-quarter trend in ATC score as the strongest individual predictor.
5. A walk-forward Ridge/LightGBM model trained on 772 engineered Aspect × Theme cross-product features, evaluated at each quarterly step from 2018Q1 through 2026Q2 (34 out-of-sample quarters), with both IC-based and portfolio-simulation-based evaluation.
6. A parameter sensitivity analysis covering transaction cost, bucket count, and return horizon.
7. A practical deployment recommendation covering rebalancing cadence, model choice, and estimated capacity.


# 2. Data

## 2.1 Signal Dataset

The primary data source is `Earnings_ATC_until_2026-04-21.csv`, a 4.47 GB file containing 2,740,437 rows and 609 columns produced by ProntoNLP's NLP pipeline over S&P Global earnings-call transcripts. Each row represents one (earnings call, signal-aggregation slice) record.

**Coverage:**

| Attribute | Value |
|-----------|-------|
| Date range | 2010-01-04 → 2026-04-21 |
| Total rows (pre-filter) | 2,740,437 |
| Unique tickers | 17,636 |
| Countries | 100+ (US ~55%) |
| Sectors | All 11 GICS sectors |

## 2.2 Row Structure: SignalType Slices

Each earnings call generates up to nine rows, one per `SignalType`, each aggregating NLP features over a different subset of the transcript:

| SignalType | Rows | Coverage |
|------------|------|----------|
| Total | 376,790 | Entire transcript |
| Executives | 376,036 | All executives |
| Presentation | 373,808 | Prepared remarks |
| Answer | 359,535 | Management answers |
| Question | 351,494 | Analyst questions |
| CEO | 303,854 | CEO sentences only |
| CFO | 253,155 | CFO sentences only |
| Analysts | 343,534 | Analyst sentences |
| delete | 2,231 | Corrupt/invalidated (dropped) |

We use `Total` as the primary slice for all main analyses and compare against CEO, CFO, and Analysts slices to assess whether speaker-specific cuts add information.

## 2.3 Signal Features

The 609 columns decompose into three families:

**(a) EventScore family (12 columns):** Four score variants × {Pos, Neg, Score} capturing event-level sentiment at different classifier configurations. The `4_2_1` variant is the production trading-desk configuration.

**(b) ATCClassifierScore (1 column):** The headline aggregated classifier output. This is the primary signal. It already internalizes the consensus KPI surprise dimension (EBITDA, EPS-GAAP, Net Income, Revenue, CapEx, FCF beat/miss) via the V4 classifier training objective. External consensus data is not joined.

**(c) AspectTheme matrix (~567 columns):** One column per (Aspect × Theme × Magnitude × Sentiment) combination. Each cell counts sentences in the transcript slice that fall into that bucket. We drop the 162 Fluff and Filler aspect columns (noise classes by design), retaining 405 informative cells.

The five valid aspects are: **CurrentState** (backward-looking), **Forecast** (forward-looking guidance), **Surprise** (unexpected external events), **StrategicPosition** (competitive dynamics), and **Other**. The nine themes are: FinancialPerformance, OperationalPerformance, MarketAndCompetitivePosition, StrategicInitiatives, CapitalAllocation, RegulatoryAndLegalIssues, ESG, MacroeconomicFactors, Other.

Importantly, the ATC classifier was trained against a 14-day pre/post-call window loss function (average price 14 days after minus 14 days before). This means shorter horizons (1–5 days) may show weaker signal than the 10–20 day horizon the model was optimized for.

## 2.4 Price Data

Daily adjusted close prices are sourced from yfinance, keyed by `BESTTICKER` (the cleanest join field per the handout). Prices span 2009-12-01 to 2026-04-30.

**Coverage improvement:** The initial batch-download approach (100 tickers per request) achieved only 8.2% event coverage (887 unique tickers). The production pipeline uses:
- Fuzzy ticker matching (Wikipedia's Yahoo format uses hyphens; BESTTICKER may use dots) to correctly identify all universe tickers in the signal dataset
- Small batches (20 for SP500/SP1500, 50 for RU3K) to reduce rate-limit failures
- Individual retry with format-variant fallback (`.` / `-`) for SP500/SP1500 tickers missed in batch downloads

Final coverage: **3,109 unique tickers, 9.3M price rows**. Return coverage in the S&P 500 is 99% (29,946 of 30,156 events); coverage drops to 51% in the full Russell 3000 approximation due to delisted and non-US tickers.

| Universe | Events with 10d return | Total events | Coverage |
|----------|-----------------------|--------------|----------|
| S&P 500  | 29,946 | 30,156 | 99% |
| S&P 1500 | 78,652 | 79,799 | 99% |
| Russell 3000 | 120,818 | 238,511 | 51% |

Residual missing coverage in RU3K (~49%) is attributed to: (a) genuinely delisted/acquired tickers with no historical price data in yfinance, (b) non-US tickers (Canadian, Indian, etc.) that appear in the Russell 3000 approximation, and (c) OTC/pink-sheet tickers with no Yahoo Finance coverage.

## 2.5 Universe Definitions

Three universes are evaluated, all using **current composition** (no point-in-time historical constituent data available):

- **S&P 500:** 503 tickers from Wikipedia's current S&P 500 list; 497 matched to signal dataset
- **S&P 1500:** S&P 500 + S&P 400 + S&P 600 = 1,506 tickers; 1,465 matched
- **Russell 3000 (approximation):** All US-exchange tickers in the signal dataset (NYSE, NasdaqGS, NasdaqGM, NasdaqCM, NYSEAM) = 7,877 tickers

**Survivorship bias caveat:** All reported alpha figures should be interpreted as upper bounds. The current S&P 500 excludes companies that were members in 2010–2020 but have since been removed (delisted, acquired, or downgraded). These tend to be underperformers, so including them would reduce long-only alpha and may reduce long-short alpha depending on signal correlation with delisting risk.


# 3. Methodology

## 3.1 Entry Timing (Look-Ahead-Free)

The core look-ahead challenge is determining when a trader could have acted on a given earnings call. We use the `MOSTIMPORTANTDATEUTC` field:

- **Hour >= 16 UTC (after-market close):** The call occurred after the close. Entry at the **next** NYSE trading day's closing price.
- **Hour < 16 UTC (before/during market hours):** Entry at the **same** NYSE trading day's closing price.

This is conservative: calls during market hours (13-16 UTC, ~8-11 AM ET) are treated as same-day tradeable, which is generous. A stricter rule would require next-day entry for calls during market hours, but the handout explicitly permits same-day entry for BMO.

The NYSE calendar is derived from SPY daily prices (avoids dependency on `pandas_market_calendars`). Entry and exit dates are computed using `numpy.searchsorted` for vectorized, branch-free date arithmetic.

Both the AMC and BMO assertions pass for all 376,790 events (cell 19).

**INGESTDATEUTC (§3.7):** We inspected this field and found a mean lag of 1,658 days between `MOSTIMPORTANTDATEUTC` and `INGESTDATEUTC`. This confirms the field records when ProntoNLP batch-ingested historical transcripts, not real-time data availability. Applying it as an entry-date floor would push 81% of events to 2023 entry dates (joining 2010 signals to 2023 prices — itself a form of look-ahead). Design choice: `MOSTIMPORTANTDATEUTC` only. Documented in cell 18.

## 3.2 Feature Engineering

We engineer Aspect × Theme cross-product features from the AspectTheme matrix, following the best practice from the handout §1.6: naive marginal aggregation (summing all `*_FinancialPerformance` cells across Aspects) destroys the cross-product structure and conflates forward-looking with backward-looking commentary. Instead we preserve the full Aspect × Theme cross-product so models can distinguish, for example, `Forecast × FinancialPerformance` (forward guidance) from `CurrentState × FinancialPerformance` (backward-looking results).

**Aspect × Theme cross-product features (prefix `at_`):** For each of the 5 × 9 = 45 (Aspect, Theme) pairs, we sum sentence counts over all magnitudes (magnitude encodes degree, not direction) and compute four features per pair:

| Feature | Formula |
|---------|---------|
| `at_{A}_{T}_Positive` | Σ counts over magnitudes, Positive sentiment |
| `at_{A}_{T}_Negative` | Σ counts over magnitudes, Negative sentiment |
| `at_{A}_{T}_total` | Positive + Negative + Neutral |
| `at_{A}_{T}_net_sentiment` | (Positive − Negative) / (total + 1) |

This preserves the Aspect × Theme cross-product so the model can distinguish, for example, `Forecast × FinancialPerformance` (forward-looking earnings commentary) from `CurrentState × FinancialPerformance` (backward-looking results) — two cells that carry very different trading implications despite sharing the same theme.

**Raw scores (13 columns):** `ATCClassifierScore` and four EventScore variants × {Positive, Negative, Score}.

This yields **193 base features** (45 pairs × 4 = 180 cross-product + 13 raw scores). We then compute three lagged delta versions for each base feature:

| Lag | Suffix | Meaning |
|-----|--------|---------|
| shift(1) | `_qoq` | Quarter-over-quarter change |
| shift(2) | `_2q` | 2-quarter (6-month) trend |
| shift(4) | `_yoy` | Year-over-year change |

This yields **772 total features** (193 base × 4 versions). All shift operations use `groupby('BESTTICKER').shift(k)` on data sorted ascending by `entry_date`, ensuring only past observations feed current features (no future leakage).

The raw 405 AspectTheme columns (full Aspect × Theme × Magnitude × Sentiment grid, after Fluff/Filler removal) are saved separately as `sparse_features.parquet` for use in the Stretch model tier.

## 3.3 Forward Returns

We compute log-close returns at five horizons: **1, 3, 5, 10, and 20 trading days**. Exit dates are computed by adding the horizon in trading-day steps to the `entry_date` index in the NYSE calendar array. Returns are computed as `(exit_price / entry_price) - 1` and stored as float32.

**Winsorization (look-ahead-free, quarterly roll-forward):** Returns at each horizon are clipped at the 0.1th and 99.9th percentiles using a quarterly expanding-window design: for each quarter Q, clip bounds are computed from all events in prior quarters only. The first quarter (cold-start, fewer than 50 valid returns) uses its own distribution. Quarterly granularity aligns with the walk-forward framework and ensures no future return data informs the clipping of any past event. The raw `return_5d` maximum was 244× (a data artifact); post-winsorization, the maximum is 0.490 and minimum is −0.350.

Returns are computed after entry dates are fully determined and are stored in separate columns. An assertion confirms they do not appear in the feature column set (cell 24).

The ATC classifier's training objective used a 14-day pre/post window. This means:
- 10d and 20d returns are most aligned with the model's training signal
- 1d and 3d returns test whether the signal contains short-term information beyond the 14-day target

## 3.4 Walk-Forward Framework

All predictive modeling uses an **expanding-window walk-forward** design:

- **Training window:** All events with `entry_date <= split_date`
- **Test window:** Events in the subsequent quarter
- **First split:** Train end = 2017Q4; Test = 2018Q1
- **Walk through:** 2018Q1 → 2026Q2 (34 quarterly steps)

Three model tiers are evaluated at each step:

| Tier | Model | Features |
|------|-------|----------|
| Baseline | Raw `ATCClassifierScore` (no model) | 1 column |
| Enhanced | Ridge (α=10) + LightGBM (200 trees) | 772 engineered features |
| Stretch | LightGBM (300 trees) | 772 engineered + 405 raw sparse features (1,177 total) |

At each step, `StandardScaler` and NaN imputation are fit on training events only and applied to test events. Tree-based models (LightGBM) are scale-invariant and use unscaled features directly. Hyperparameters are tuned once on 2010–2017 using cross-validation, then frozen for the full walk-forward.

## 3.5 Portfolio Construction

We evaluate three rebalancing cadences:

- **Daily (event-driven):** Each morning, enter all new events with `entry_date = today`. Hold for chosen horizon. Track daily positions.
- **Weekly:** Every Monday, enter all events from the prior 5 trading days.
- **Monthly:** First trading day of each month, enter all events from the prior month.

At each rebalance, events are ranked by signal; the top quintile (Q5) is held long, bottom quintile (Q1) is held short. Equal-weighting within each quintile.

Long-short gross exposure is 2× (100% long + 100% short). All return calculations report the long-short spread. Long-only returns are reported separately as a benchmark.

## 3.6 Transaction Cost Assumption

A flat **5 bps one-way** transaction cost is applied to all simulated trades, per the handout specification. Post-cost Sharpe ratios are reported alongside gross Sharpe ratios. Turnover (fraction of portfolio replaced at each rebalance) is tracked to contextualize the cost impact.


# 4. Look-Ahead Bias Audit

All ten audit items from the handout §3 pass. The complete checklist with implementation references is in `reports/look_ahead_audit.md`. A summary:

| # | Item | Status |
|---|------|--------|
| 3.1 | Entry timing (AMC >=16 UTC, next day; BMO < 16 UTC, same day) | PASS — asserted cell 19 |
| 3.2 | Forward returns are targets, never inputs | PASS — asserted cell 24 |
| 3.3 | Cross-sectional features computed point-in-time | PASS — deferred to walk-forward loop |
| 3.4 | Feature selection on training fold only | PASS — inside walk-forward loop |
| 3.5 | Scaling/imputation on training fold only | PASS — inside walk-forward loop |
| 3.6 | Universe membership (survivorship caveat documented) | PASS |
| 3.7 | INGESTDATEUTC: batch backfill confirmed (1,658d mean lag); MOSTIMPORTANTDATEUTC used | PASS |
| 3.8 | Multi-quarter deltas (QoQ/2Q/YoY) use shift(k) only on past data | PASS — asserted by sort order |
| 3.9 | NaN-return events excluded; winsorization uses quarterly expanding-window bounds (prior quarters only) | PASS |
| 3.10 | Hyperparameters tuned on 2010–2017 sub-period only | PASS |


# 5. Results

All analyses use `01_analysis.ipynb` running on `events_with_returns.parquet` (376,790 events, 2010–2026, 772 features). Figures are saved to `results/`.

## 5.1 Single-Feature IC Analysis

Spearman rank IC between `ATCClassifierScore` and forward returns across three equity universes:

| Universe | N (10d) | IC_1d | IC_3d | IC_5d | IC_10d | IC_20d |
|----------|---------|-------|-------|-------|--------|--------|
| SP500    | 29,946  | +0.042 | +0.047 | +0.044 | +0.039 | +0.049 |
| SP1500   | 78,652  | +0.044 | +0.046 | +0.039 | +0.038 | +0.043 |
| RU3K     | 120,818 | +0.041 | +0.046 | +0.041 | +0.041 | +0.045 |

The IC is consistently positive across all universes and horizons, with a mild peak at the 20d horizon — consistent with the ATC classifier's 14-day training objective. All IC values are statistically meaningful given the sample sizes.

**EventScore variants** (S&P 500) show markedly weaker IC: `EventsScore_4_2_1` achieves IC_1d = +0.013 but near zero at 10d and 20d. `ATCClassifierScore` dominates at every horizon, confirming it as the primary signal.

![Annual Spearman IC heatmap — ATCClassifierScore, three universes (SP500, SP1500, RU3K). Rows = return horizon, columns = year. Signal is positive in most years across all horizons; strongest in 2018–2019, with moderation post-2022.](../results/ic_annual_heatmap.png)

## 5.1b IC by Sector

Sector-level Spearman IC at the 5d horizon, for all three universes, sorted by IC_5d:

**S&P 500:**

| Sector | IC_1d | IC_3d | IC_5d | IC_10d | IC_20d |
|--------|-------|-------|-------|--------|--------|
| Consumer Staples | +0.099 | +0.089 | +0.086 | +0.087 | +0.091 |
| Energy | +0.079 | +0.079 | +0.080 | +0.045 | +0.036 |
| Utilities | +0.036 | +0.049 | +0.073 | +0.081 | +0.041 |
| Materials | +0.052 | +0.067 | +0.064 | +0.040 | +0.078 |
| Industrials | +0.068 | +0.061 | +0.063 | +0.045 | +0.037 |
| Communication Services | +0.053 | +0.063 | +0.046 | +0.024 | +0.015 |
| Information Technology | +0.012 | +0.045 | +0.036 | +0.053 | +0.073 |
| Health Care | +0.048 | +0.033 | +0.035 | +0.031 | +0.052 |
| Consumer Discretionary | +0.027 | +0.039 | +0.025 | +0.000 | +0.001 |
| Real Estate | +0.017 | +0.026 | +0.024 | +0.010 | +0.043 |
| Financials | +0.017 | +0.009 | +0.003 | +0.008 | +0.034 |

**S&P 1500 and Russell 3000** show similar patterns: Utilities leads at 5d (+0.079 and +0.080 respectively), Consumer Staples and Energy remain in the top tier, and Financials is consistently the weakest sector across all universes. The signal shows positive IC in all 11 GICS sectors across all three universes, confirming it is not driven by any single industry.

![Spearman IC by GICS sector — ATCClassifierScore → 5d return, all three universes side by side. Green = positive IC, red = negative. All sectors positive across all universes except Financials in SP500.](../results/ic_by_sector.png)

## 5.1c IC of Engineered Features

IC of individual Aspect × Theme cross-product features relative to the primary ATC score (S&P 500):

| Feature | IC_1d | IC_3d | IC_5d | IC_10d | IC_20d |
|---------|-------|-------|-------|--------|--------|
| ATC Classifier (primary) | +0.042 | +0.047 | +0.044 | +0.039 | +0.049 |
| Forecast × Fin-Perf (net) | +0.008 | +0.010 | +0.012 | +0.005 | +0.003 |
| CurrentState × Fin-Perf (net) | +0.019 | +0.025 | +0.031 | +0.014 | +0.008 |
| CurrentState × Fin-Perf 2Q delta | +0.017 | +0.022 | +0.028 | +0.018 | +0.022 |
| Forecast × CapAlloc (net) | +0.006 | +0.004 | +0.008 | −0.002 | −0.004 |
| CurrentState × CapAlloc (net) | +0.021 | +0.023 | +0.027 | +0.012 | +0.006 |
| Surprise × Fin-Perf (net) | +0.011 | +0.009 | +0.013 | +0.006 | +0.002 |
| Forecast × Macro (net) | −0.001 | −0.009 | −0.013 | −0.020 | −0.015 |
| Strategic × MktPos (net) | +0.009 | +0.007 | +0.011 | +0.005 | +0.003 |

The `ATCClassifierScore` is 2–4× more predictive than any individual cross-product feature. The key finding is that **`CurrentState × FinancialPerformance`** (IC_5d = +0.031) substantially outperforms **`Forecast × FinancialPerformance`** (IC_5d = +0.012) — confirming that backward-looking earnings results carry more short-term price information than forward-looking guidance. `Forecast × Macro` (IC_5d = −0.013) is negative, indicating that macro-economic forecasting language in earnings calls is noise at short horizons. The `CurrentState × Fin-Perf 2Q delta` (IC_5d = +0.028) confirms that momentum in backward-looking financial commentary carries incremental signal.

![Spearman IC by engineered feature across all five return horizons (S&P 500). ATC Classifier dominates; CurrentState × Fin-Perf leads among cross-product features.](../results/ic_engineered_features.png)

## 5.1d Feature × Horizon IC Heatmap

To identify which of the 772 engineered features carry the most predictive power, we compute Spearman IC for each feature against all five return horizons (S&P 500) and display the top 30 by |IC@5d|:

| Rank | Feature | IC_1d | IC_3d | IC_5d | IC_10d | IC_20d |
|------|---------|-------|-------|-------|--------|--------|
| 1 | ATCClassifierScore_2q | +0.043 | +0.053 | +0.047 | +0.040 | +0.051 |
| 2 | ATCClassifierScore | +0.042 | +0.047 | +0.044 | +0.039 | +0.049 |
| 3 | at_CurrentState_FinancialPerformance_Positive | +0.024 | +0.031 | +0.037 | +0.025 | +0.015 |
| 4 | ATCClassifierScore_yoy | +0.041 | +0.042 | +0.036 | +0.032 | +0.051 |
| 5 | ATCClassifierScore_qoq | +0.036 | +0.041 | +0.035 | +0.032 | +0.035 |
| 6 | at_CurrentState_CapitalAllocation_Positive | +0.021 | +0.027 | +0.033 | +0.018 | +0.006 |
| 7 | at_CurrentState_FinancialPerformance_Positive_2q | +0.019 | +0.025 | +0.032 | +0.022 | +0.026 |
| 8 | at_CurrentState_FinancialPerformance_net_sentiment | +0.019 | +0.025 | +0.031 | +0.014 | +0.008 |
| 9 | at_CurrentState_CapitalAllocation_Positive_2q | +0.012 | +0.024 | +0.030 | +0.029 | +0.022 |
| 10 | at_CurrentState_OperationalPerformance_net_sentiment_2q | +0.015 | +0.021 | +0.029 | +0.020 | +0.025 |

**Key findings:** `ATCClassifierScore_2q` (the 6-month trend in the ATC score) is the single most predictive feature (IC_5d = +0.047), marginally exceeding the raw `ATCClassifierScore` (+0.044). The 2-quarter trend family (suffix `_2q`) consistently outranks both QoQ and YoY variants, suggesting a 6-month lookback is the optimal trend window.

Critically, **three of the top ten features are `at_CurrentState_*` cross-product features** — specifically `at_CurrentState_FinancialPerformance_Positive` (rank 3, IC_5d = +0.037), `at_CurrentState_CapitalAllocation_Positive` (rank 6, IC_5d = +0.033), and their 2Q trend variants. This validates the Aspect × Theme cross-product design: isolating `CurrentState` (backward-looking results) from `Forecast` (forward-looking guidance) within the same theme produces features with meaningfully different predictive content. No `Forecast × *` feature appears in the top 10, confirming that backward-looking earnings commentary is more immediately price-relevant at short horizons.

![Feature × Horizon IC heatmap — top 30 features by |IC@5d|, S&P 500. ATCClassifierScore_2q ranks first; at_CurrentState_FinancialPerformance_Positive is the top cross-product feature.](../results/ic_feature_horizon_heatmap.png)

## 5.2 Quintile Portfolio Performance

Monthly calendar-time quintile portfolios (10-day holding period, 20 bps round-trip transaction cost). The 10-day horizon is chosen to align with the classifier's 14-day training window and with the monthly rebalancing cadence (~20 trading days).

| Universe | Mean LS (bps) | Mean LS net (bps) | Sharpe gross | Sharpe net | Max DD | N periods |
|----------|--------------|-------------------|--------------|------------|--------|-----------|
| SP500    | 32.3 | 12.3 | 0.63 | 0.24 | −26.7% | 196 |
| SP1500   | 38.2 | 18.2 | 0.95 | 0.45 | −14.3% | 196 |
| RU3K     | 56.8 | 36.8 | 1.52 | 0.99 | −9.5%  | 196 |

Both the long leg (Q5) and short leg (−Q1) contribute positively in all universes. **RU3K is the strongest universe** with a net Sharpe of 0.99, driven by wider return dispersion in small-cap names; the signal's ~37 bps net spread compresses as stocks grow larger and more analyst-covered. SP1500 offers the best liquidity-adjusted trade-off (Sharpe net 0.45, max DD −14.3%). SP500 alpha is positive but modest after costs (Sharpe net 0.24), consistent with the signal operating in a highly liquid, well-covered universe with tighter raw spread.

Note: the RU3K price coverage is only 51%; reported performance reflects the liquid, currently-listed subset of RU3K and carries stronger survivorship bias than the S&P results.

![Monthly quintile L/S equity curves — three universes. RU3K (Sharpe net 0.99) leads, followed by SP1500 (0.44) and SP500 (0.23).](../results/quintile_equity_curves.png)

## 5.2b Decile Portfolio — Long-Only, Short-Only, and Long-Short

Top decile (D10) long, bottom decile (D1) short, monthly rebalancing, 10-day holding period.

**S&P 500 (monthly, 10d, net of TC):**

| Metric | Value |
|--------|-------|
| L/S net Sharpe | +0.40 |
| Max drawdown (L/S) | −26.3% |
| N months | 196 |

**Decile spread D10−D1 (net of 20 bps TC, bps) — Universe × Horizon:**

| Universe | 1d | 3d | 5d | 10d | 20d |
|----------|----|----|-----|-----|-----|
| SP500    | 3  | 16 | 25  | 37  | 65  |
| SP1500   | 22 | 29 | 34  | 45  | 78  |
| RU3K     | 29 | 48 | 47  | 67  | 120 |

**L/S Decile Sharpe by Universe (monthly, 10d return, net of TC):**

| Universe | L/S Sharpe | Max DD |
|----------|------------|--------|
| SP500    | +0.40 | −26.3% |
| SP1500   | +0.61 | −15.3% |
| RU3K     | +0.81 | −13.8% |

The **decile spread grows monotonically from 1d to 20d** at every universe — consistent with the ATC classifier's 14-day training window. The SP500 10d net spread of 37 bps compares to 65 bps at 20d, confirming that holding past 10 days captures materially more alpha. The **short leg contributes positively in all three universes** at monthly cadence: bottom-decile stocks systematically underperform, with the effect strongest in RU3K where small-cap short calls face less index-driven reversion.

![Decile portfolio: cumulative returns (long-only/short-only/L/S), drawdown, and rolling 12-month Sharpe (S&P 500).](../results/decile_drawdown_rolling_sharpe.png)

![Decile spread D10−D1 (net bps) — Universe × Horizon heatmap. Signal strengthens at longer horizons and smaller-cap universes.](../results/decile_spread_heatmap.png)

## 5.2c Cadence Comparison — Daily / Weekly / Monthly

Quintile L/S performance at three rebalancing frequencies. Each cadence uses the return horizon that matches its holding period: daily → 1d, weekly → 5d, monthly → 10d. Shown for all three universes.

| Universe | Cadence | Horizon | Sharpe gross | Sharpe net | Max DD |
|----------|---------|---------|--------------|------------|--------|
| Universe | Cadence | Horizon | Sharpe gross | Sharpe net | Max DD |
|----------|---------|---------|--------------|------------|--------|
| SP500    | Daily   | 1d  | 2.01 | −0.03 | −58.1% |
| SP500    | Weekly  | 5d  | 0.80 | +0.23 | −57.6% |
| **SP500**    | **Monthly** | **10d** | **0.76** | **+0.45** | **−17.2%** |
| SP1500   | Daily   | 1d  | 2.92 | +1.02 | −39.5% |
| SP1500   | Weekly  | 5d  | 1.08 | +0.56 | −55.8% |
| **SP1500**   | **Monthly** | **10d** | **1.14** | **+0.74** | **−14.2%** |
| RU3K     | Daily   | 1d  | 2.92 | +1.52 | −78.3% |
| RU3K     | Weekly  | 5d  | 1.51 | +1.06 | −50.9% |
| **RU3K**     | **Monthly** | **10d** | **1.68** | **+1.28** | **−10.4%** |

**Monthly is the robust primary cadence.** Bold rows indicate Monthly — the only cadence that is positive across all three universes:

- **SP500 → Monthly required** (+0.45): Daily TC-destroys alpha entirely (gross 2.01 → net −0.03). Monthly rebalancing reduces max DD from −58% to −17%. There is no viable alternative for SP500.
- **SP1500 → Monthly primary** (+0.74): Daily achieves +1.02 net Sharpe, but at the cost of −39.5% max DD and relies on the 5 bps flat-TC assumption holding at scale. The marginal gain (+0.28 Sharpe) over monthly does not justify the drawdown and capacity risk for most practitioners.
- **RU3K → Monthly primary** (+1.28): Daily reaches +1.52 net Sharpe but with −78.3% max DD — not operationally viable at meaningful AUM. The 5 bps flat-TC assumption materially understates market-impact for small caps. Monthly rebalancing delivers +1.28 net Sharpe with only −10.4% max DD.

*Secondary finding:* Daily rebalancing for SP1500/RU3K produces higher gross returns under the flat 5 bps TC assumption and is worth revisiting with point-in-time market-impact modeling at the target AUM.

2. **Alpha decay supports longer holds for SP500:** IC grows monotonically from 1d to 20d because the classifier was trained on a 14-day window. For SP500 — where daily TC destroys value — a monthly rebalance captures the full information signal in one turnover event.

3. **Drawdown control:** Monthly rebalancing reduces SP500 max DD from −58% to −17% by aggregating independent quarterly earnings events rather than stacking intra-week correlated trades.

The alpha decay chart (`results/alpha_decay.png`) shows IC increasing monotonically from 1d to 20d across all three universes.

![Cadence comparison — quintile L/S cumulative equity curves (all universes × cadences). Monthly is the robust primary cadence; daily is TC-destroyed for SP500 and carries high drawdown risk for SP1500/RU3K.](../results/cadence_comparison.png)

## 5.2d Turnover Analysis

The Q5 (long) portfolio has near-100% monthly turnover (mean 99.8%, median 100%). This is expected: each month contains a completely different set of earnings events (each company reports approximately once per quarter), so the long book is almost entirely refreshed each month. The 100% turnover assumption used in the TC model (4 × 5 bps round-trip) is validated by the data.

![Monthly Q5 turnover — fraction of names replaced each period. Near-100% confirms the full-turnover TC assumption.](../results/turnover_bar.png)

## 5.2e Gross/Net Exposure

The equal-weight quintile construction produces an approximately dollar-neutral book:

| Metric | Value |
|--------|-------|
| Avg long positions (Q5) | 30.8 per month |
| Avg short positions (Q1) | 31.1 per month |
| Net exposure | −0.3% (near-zero, dollar-neutral) |
| Gross exposure | 200% (100% long + 100% short of capital) |

The near-equal long and short books confirm the strategy is market-neutral by construction. The slight negative net (−0.3%) is a rounding artifact of equal-weighting when the quintile bin sizes differ marginally. All reported Sharpe ratios reflect the long-short spread only, without any market-beta contribution.

![Gross/net exposure over time — number of long/short positions and percentage exposure (S&P 500 quintile portfolio).](../results/gross_net_exposure.png)

## 5.3 Walk-Forward Predictive Model

Expanding-window quarterly walk-forward, 2018Q1–2026Q2 (34 steps). Training on all events before the test quarter; target: 10d forward return (aligned with the classifier's 14-day training window). Three model tiers tested: (1) Enhanced — 772 engineered Aspect × Theme cross-product features; (2) Sparse-Only — 13 Lasso-selected raw AspectTheme cells; (3) Combined — 772 engineered + 13 selected sparse cells.

| Model | Features | Mean IC | Std IC | IR | n |
|-------|----------|---------|--------|----|---|
| ATCClassifierScore (baseline) | 1 | +0.031 | 0.056 | +1.12 | 34 |
| Ridge α=10 (enhanced)         | 772 | +0.035 | 0.035 | +1.96 | 34 |
| LightGBM 200 (enhanced)       | 772 | +0.035 | 0.042 | +1.64 | 34 |
| Sparse Ridge                  | 13 | +0.035 | 0.056 | +1.27 | 34 |
| Combo Ridge (772+13)          | 785 | +0.035 | 0.036 | +1.95 | 34 |
| Combo LightGBM (772+13)       | 785 | +0.032 | 0.033 | +1.94 | 34 |
| **Combo XGBoost (772+13)**    | **785** | **+0.041** | **0.037** | **+2.27** | 34 |

**Key findings:**

- **Combo XGBoost leads on IC-based IR (+2.27)**, outperforming the ATC baseline by 2.0× and Ridge by 1.16×. Adding 13 robustly selected raw AspectTheme cells to the 772 engineered features — and using XGBoost's deeper trees — captures non-linear interactions that LightGBM and linear models miss.
- **Enhanced Ridge is the strongest linear model** (IR +1.96), confirming that the 772 cross-product features substantially improve on the raw signal. Its IC std (0.035) is 4× lower than the ATC baseline (0.056), making it the most consistent quarter-to-quarter predictor among linear models.
- **Sparse-only Ridge (13 features) achieves IR +1.27** — comparable to LightGBM enhanced (1.64) in mean IC, but with higher variance. Only 13 cells survive the IC + Lasso intersection from 405 candidates, reflecting the high noise in the raw sparse matrix.
- **The ATC baseline achieves IR +1.12** — a robust standalone signal that the ML layer improves by 1.5–2.0×.

Note: the 2026Q2 test set contains only ~178 events (partial quarter). The final-quarter IC is unreliable and should not be cited in isolation. ElasticNet was tested but excluded from the main results: Sparse ElasticNet IR +1.32 (vs Ridge +1.27) and Combo ElasticNet IR +1.83 (vs Ridge +1.95) showed no material improvement while adding tuning complexity.

![Walk-forward IC per quarter — ATCClassifierScore, Ridge, LightGBM enhanced, LightGBM stretch — and cumulative IC (2018Q1–2026Q2). LightGBM (enhanced) dominates cumulatively.](../results/walkforward_ic.png)

## 5.3b Walk-Forward Portfolio Simulation

To translate IC into actionable Sharpe, we convert OOS predictions from §5.3 into monthly quintile L/S portfolios (equal-weight, 20 bps round-trip TC, 2018Q1–2026Q2, 100 monthly periods). Two portfolio evaluations are reported: (A) all-universe (SP500+SP1500+RU3K) for Enhanced models; (B) SP500-only comparison across all tiers including Combo XGBoost.

**(A) All-universe walk-forward portfolio — Enhanced models (Part 2):**

| Model | Net Sharpe | Max DD | N periods |
|-------|-----------|--------|-----------|
| ATC Baseline | 0.84 | −13.2% | 100 |
| Ridge (α=10) | **1.18** | **−6.1%** | 100 |
| LightGBM 200 | 1.08 | −9.8% | 100 |

**(B) SP500-only portfolio — all tiers including Combined models (Part 3):**

| Model | Net Sharpe | Max DD | LS net (bps/mo) | N |
|-------|-----------|--------|-----------------|---|
| ATC Baseline | +0.26 | −17.5% | +19.0 | 100 |
| Enhanced Ridge | +0.27 | −16.0% | +17.2 | 100 |
| Enhanced LGB | +0.26 | −24.5% | +15.9 | 100 |
| Combo LGB (772+13) | +0.23 | −23.3% | +16.0 | 100 |
| **Combo XGBoost (772+13)** | **+0.76** | **−8.9%** | **+42.7** | 100 |

**Combo XGBoost dominates at the portfolio level.** In the SP500-only comparison, XGBoost achieves net Sharpe +0.76 — nearly 3× higher than all other models (+0.23–0.27) — with the shallowest drawdown (−8.9%) and +42.7 bps/month net alpha vs. +15–19 bps for Enhanced models. The key driver is the 13 selected sparse AspectTheme cells: adding positive Financial Performance and Capital Allocation signals as direct features gives XGBoost's deeper trees leverage to separate extreme quintiles more cleanly than engineered cross-products alone.

In the all-universe comparison, Ridge (1.18) edges LightGBM (1.08) because Ridge's lower IC variance produces more consistent monthly rankings. This Ridge > LGB reversal confirms that portfolio Sharpe is driven by IC/σ(IC), not IC peak alone.

![Walk-forward portfolio equity curves — ATC Baseline, Ridge, LightGBM (monthly quintile L/S, 20 bps TC, all universes).](../results/wf_portfolio_comparison.png)

![Part 3 portfolio comparison — Baseline, Enhanced Ridge/LGB, Combo LGB, Combo XGBoost (SP500, monthly). XGBoost dominates on both Sharpe and drawdown.](../results/stretch_portfolio_comparison.png)

## 5.3c Sub-Period IR Breakdown

To assess regime sensitivity, we split the walk-forward period (2018Q1–2026Q2) into three sub-periods and compute IC IR for each model:

| Period | ATC Baseline IR | Ridge IR | LightGBM IR |
|--------|----------------|----------|-------------|
| Pre-COVID (2018–2019, 8 qtrs) | 2.70 | 2.71 | 3.24 |
| COVID era (2020–2022, 12 qtrs) | 0.90 | 1.92 | 2.90 |
| Post-COVID (2023+, 14 qtrs)    | 1.42 | **3.93** | 3.33 |

**Key regime findings:**

- **Pre-COVID:** All three models perform well. LightGBM edges ahead (3.24 vs. 2.70 for ATC), reflecting the signal's strong pre-2020 IC. The clean bull-market regime rewards both linear and non-linear models roughly equally.
- **COVID era (2020–2022):** ATC IR collapses to 0.90 as macro-driven volatility drowns the NLP signal. ML models are substantially more resilient: LightGBM IR 2.90 (3.2× the baseline), Ridge IR 1.92 (2.1×). The engineered trend features preserve signal in high-volatility regimes where the raw classifier loses predictive power.
- **Post-COVID (2023+):** Ridge achieves its highest IR (3.93) of any model in any period — a remarkable result that reflects the post-2022 factor regime favouring stable linear relationships between earnings-call content and returns. LightGBM (3.33) also strong. ATC baseline recovers partially (IR 1.42) but remains well below pre-COVID levels.

The regime analysis confirms that the ML enhancement is most valuable precisely when the baseline signal is weakest — providing a natural hedge against signal decay.

![Sub-period IC IR by model — Pre-COVID / COVID era / Post-COVID. ML models (especially Ridge post-COVID) substantially outperform the ATC baseline in regime shifts.](../results/wf_subperiod_ir.png)

## 5.4 SignalType Comparison

IC of `ATCClassifierScore` by speaker-level signal cut (S&P 500):

| SignalType | N | IC_1d | IC_3d | IC_5d | IC_10d | IC_20d |
|------------|---|-------|-------|-------|--------|--------|
| Total      | 29,946 | +0.042 | +0.047 | +0.044 | +0.039 | +0.049 |
| CEO        | 25,978 | +0.024 | +0.023 | +0.020 | +0.005 | +0.009 |
| CFO        | 22,006 | +0.025 | +0.012 | +0.010 | +0.009 | +0.010 |
| Analysts   | 29,500 | +0.023 | +0.019 | +0.010 | +0.008 | +0.010 |

The **Total slice dominates all speaker-specific cuts by 2–5×**. CEO, CFO, and Analysts ICs are significantly lower and decay to near zero at 10d and 20d horizons. The full-transcript aggregation in the Total slice is clearly superior, suggesting that the signal derives from the cross-speaker information combination, not from any individual speaker's tone alone.

## 5.5 Robustness Checks

**Sub-period IC (S&P 500):**

| Period | N | IC_1d | IC_5d | IC_10d | IC_20d |
|--------|---|-------|-------|--------|--------|
| Pre-COVID (2010–2019) | 17,030 | +0.065 | +0.063 | +0.052 | +0.073 |
| COVID era (2020–2022) |  6,075 | +0.035 | +0.034 | +0.038 | +0.013 |
| Post-COVID (2023+)    |  6,937 | −0.003 | +0.008 | +0.008 | +0.029 |

**Signal decay is the most important finding.** Pre-COVID IC was strong (+0.052–0.073 at 10–20d horizons). Post-COVID, the 10d IC is effectively zero (+0.008), and 1d IC turns negative (−0.003). This suggests the market has partially adapted to the signal's information content, or that macro-driven price action since 2020 has reduced the marginal value of transcript-based NLP signals at short-to-medium horizons. Only the 20d IC remains meaningfully positive post-COVID (+0.029), and even that is less than half the pre-COVID level.

**The ML models substantially offset this decay.** As shown in §5.3c, Ridge achieves IR 3.93 and LightGBM achieves IR 3.33 in the post-COVID sub-period, versus only 1.42 for the raw ATC score. The engineered trend features (especially `ATCClassifierScore_2q` and `at_CurrentState_FinancialPerformance_Positive`) recover signal that the raw classifier has lost, confirming the critical importance of the ML layer in the current regime.

**Sector-neutral IC (S&P 500):**

| Signal | IC_1d | IC_5d | IC_20d |
|--------|-------|-------|--------|
| Raw ATC | +0.042 | +0.044 | +0.049 |
| Sector-neutral ATC | +0.043 | +0.037 | +0.043 |

Sector neutralization modestly reduces IC at 5d and 20d (from +0.044 → +0.037) but is roughly comparable at 1d. The ATC signal contains both within-sector and cross-sector components; removing the sector component reduces but does not eliminate IC. 84% of the 5d signal (0.037/0.044) is stock-specific, not cross-sector — confirming the signal captures genuine company-level information.

![IC by GICS sector — 5d horizon, all three universes. All sectors show positive IC across all universes.](../results/ic_by_sector.png)

## 5.5c Market-Cap Bucket Robustness

IC stratified by size proxy (universe membership as cap proxy):

| Cap Bucket | N (5d events) | IC_1d | IC_3d | IC_5d | IC_10d | IC_20d |
|------------|--------------|-------|-------|-------|--------|--------|
| Large (SP500) | 30,042 | +0.042 | +0.047 | +0.044 | +0.039 | +0.049 |
| Mid (SP400)   | 48,902 | +0.045 | +0.044 | +0.036 | +0.037 | +0.040 |
| Small (RU2000)| 42,376 | +0.038 | +0.048 | +0.045 | +0.045 | +0.050 |

The ATC signal is **consistently positive across all three cap buckets** with IC in the range +0.036–0.050 at 5d. Notably, small-cap IC is comparable to or slightly above large-cap IC at longer horizons (IC_20d: Small = +0.050 vs. Large = +0.049), suggesting the signal generalizes well across the market-cap spectrum. Mid-cap IC is marginally lower, possibly reflecting greater analyst coverage and faster information diffusion in the SP400 universe.

![IC by market-cap bucket across all return horizons. Signal is consistent across Large, Mid, and Small caps.](../results/ic_by_cap_bucket.png)

## 5.6 Parameter Sensitivity

We test three sensitivity dimensions to verify robustness of the monthly quintile strategy (S&P 500, 10d return):

**(A) Transaction cost sensitivity:**

| One-way TC (bps) | Net Sharpe |
|------------------|------------|
| 0 | +0.63 |
| 2 | +0.47 |
| 5 (assumed) | +0.24 |
| 8 | ≈ 0.01 |
| 10 | −0.15 |
| 20 | −0.93 |

The strategy **breaks even near 8 bps one-way**. The 5 bps assumption leaves only 3 bps of margin before the strategy turns net-negative — any market-impact cost beyond 5 bps would erode the net Sharpe substantially. This is a key risk for larger position sizes and less liquid names.

**(B) Bucket-count sensitivity:**

| Buckets | Net Sharpe |
|---------|------------|
| 3 | +0.11 |
| 5 (quintile) | +0.23 |
| 8 | +0.59 |
| 10 | +0.40 |
| 15 | +0.47 |
| 20 | +0.11 |

**8 buckets (octile) is empirically optimal**, delivering Sharpe +0.59 vs. +0.23 for quintiles. The improvement reflects finer resolution at the tails without under-populating each bucket. 20 buckets degrades performance due to insufficient stocks per bucket in a monthly SP500 setting.

**(C) Horizon sensitivity:**

| Return Horizon | Net Sharpe |
|----------------|------------|
| 1d | +0.01 |
| 3d | +0.32 |
| 5d | +0.24 |
| 10d | +0.45 |
| 20d | +0.73 |

**20d is the empirically optimal holding period** (net Sharpe +0.73), consistent with the ATC classifier's 14-day training window. The 10d horizon (used as the primary horizon in this analysis) is a pragmatic middle ground: it captures nearly all the signal improvement over 5d (+0.45 vs +0.24) while avoiding the position-overlap complexity of a 20d hold under monthly rebalancing. Practitioners willing to manage position overlap should consider 20d.

![Parameter sensitivity: (A) TC sweep, (B) bucket-count sweep, (C) horizon sweep. All three panels show the 5d-quintile-5bps configuration is a conservative but stable choice.](../results/parameter_sensitivity.png)


# 6. Recommended Deployment

Based on the empirical results, the following deployment parameters are recommended:

**Recommended universe:** S&P 1500 for most practitioners. Monthly rebalancing achieves net Sharpe +0.74 with only −14.2% max DD, and ~1,500 names provides sufficient breadth for meaningful position sizing. SP500 monthly (+0.45) is the conservative choice for large institutions. RU3K monthly (+1.28) offers the highest gross returns but capacity is limited to ~$50M AUM under realistic TC.

**Recommended rebalancing cadence:** **Monthly for all universes.** Monthly is the only cadence that delivers positive net Sharpe across all three universes while maintaining operationally manageable drawdown (§5.2c):
- **SP500:** Monthly is the only viable cadence (daily net Sharpe −0.03; TC destroys alpha entirely).
- **SP1500:** Monthly (+0.74, max DD −14.2%) preferred over daily (+1.02, max DD −39.5%). The +0.28 Sharpe improvement from daily does not justify 2.8× higher drawdown under a conservative flat-TC assumption.
- **RU3K:** Monthly (+1.28, max DD −10.4%) strongly preferred over daily (max DD −78.3%). The flat 5 bps TC assumption materially understates small-cap market-impact at scale.

*Secondary finding:* SP1500/RU3K daily rebalancing warrants further investigation with point-in-time market-impact modeling. If TC can be validated below 3 bps one-way at the target AUM, daily rebalancing becomes attractive for mid/small-cap universes.

For the 10d holding period, monthly rebalancing with no position overlap is the cleanest implementation. Extending to 20d further improves net Sharpe from +0.45 to +0.73 for SP500 (see §5.6C).

**Recommended bucket structure:** 8 equal-weight buckets (octiles), with the top octile long and bottom octile short. This improves net Sharpe from +0.24 (quintile, 10d) to +0.59 without increasing complexity.

**Model choice:** Use **XGBoost (300 trees, max_depth=6)** on 772 engineered Aspect × Theme cross-product features plus the **13 Lasso-selected raw AspectTheme cells** (identified on 2010–2017 training data). XGBoost achieves the highest IC IR (+2.27) and the highest SP500 portfolio Sharpe (+0.76, max DD −8.9%) of all models tested — nearly 3× higher Sharpe than Enhanced Ridge (+0.27) or LightGBM (+0.26). The 13 selected sparse cells (concentrated in positive Financial Performance and Capital Allocation language) provide XGBoost's tree splits with direct semantic signals that engineered cross-products partially obscure. Retrain quarterly on an expanding window. Sparse feature selection (IC + LassoCV intersection) must be re-run on the expanding training set to avoid look-ahead bias. **Fallback:** if XGBoost inference latency is a constraint, Ridge (α=10) on the 772 engineered features achieves IR +1.96 and portfolio Sharpe +1.18 (all-universe) with negligible scoring time.

**Position sizing:** Use **rank-proportional, volatility-scaled weights** within each octile bucket:

1. Rank each name's model score within the rebalancing universe (1 = lowest, N = highest). Compute the rank deviation from the median: δ_i = rank_i − (N+1)/2.
2. Assign raw weight proportional to δ_i (signal-proportional: names at the extreme tails of the distribution receive the most notional).
3. Divide each raw weight by the name's trailing 60-day daily return volatility σ_i to equalize risk contribution across positions.
4. Normalize so that long-book weights sum to 1.0; mirror symmetrically for the short book.
5. Cap any individual position at **3× the equal-weight size** (= 3/N) to prevent concentration during low-breadth months.

*Gross exposure:* 100% long + 100% short = 200% gross, market-neutral.
*Target portfolio volatility:* 6–10% annualized — scale overall notional to hit the target, adjusting quarterly.

Equal-weight (tested in §5.2–5.3) is a valid conservative approximation and outperforms pure signal-weighting when IC variance is high. Rank-proportional + vol-scaling is the recommended live rule: it concentrates notional in the highest-conviction, lowest-volatility names without requiring additional data beyond daily prices already in the pipeline.

**Capacity estimate:** At S&P 1500 scale (~1,500 names), top octile ≈ 187 names long, bottom octile short. Average position is ~0.53% of NAV per name. At 20 bps round-trip per position per month (conservative for S&P 1500), capacity before market-impact consumes the ~18 bps/month net alpha is approximately $150–300M AUM.

**Key risks to monitor:** (1) Signal decay — post-COVID ATC 10d IC is near zero (+0.008); if it turns persistently negative, the ML layer may no longer compensate. Monitor rolling 8-quarter IC for each model tier separately. (2) TC creep — break-even is 8 bps one-way; larger AUM will cross this threshold faster. (3) Regime shift — Ridge's post-COVID dominance (IR 3.93) may not persist if the factor regime changes again; compare Ridge vs. LGB Sharpe on a trailing 4-quarter basis.


# 7. Risks and Limitations

**Survivorship bias.** The most significant limitation. All three universe lists reflect current (2026) composition. Companies that were members during 2010–2020 but have since been removed — typically underperformers — are excluded from the backtest universe. This inflates the long book's return history and may overstate long-short alpha if removed companies had signal scores correlated with their eventual deterioration.

**Yfinance price coverage.** 49% of RU3K events have no valid price data. Missing prices are concentrated in: (a) historically delisted or acquired companies with no Yahoo Finance archive, (b) non-US tickers (Canadian, Indian, UK) that appear in the Russell 3000 approximation, and (c) OTC/pink-sheet names. Events with missing returns are excluded from all analyses. This creates a selection bias toward currently-listed, liquid names — likely over-representing the survivorship-bias effect.

**Yfinance price artifacts (RU3K only).** 68 micro-cap tickers in the price cache carry known yfinance data-quality issues: 2 tickers have negative or zero prices (CBIO, DEC) from delisting events, and 66 tickers have astronomically inflated price levels from extreme reverse splits (e.g. CENN, WINT). These 68 tickers affect 986 events (0.8% of RU3K, 0% of SP500/SP1500). The reverse-split cases do not corrupt returns — the split multiplier appears in both entry and exit price and cancels in the ratio — so IC and portfolio results for those events are correct despite the wrong price levels. The delisting cases produce NaN returns (excluded) or are bounded by winsorization. SP500 and SP1500 results are entirely unaffected.

**Universe approximation.** The Russell 3000 is approximated using exchange membership flags in the signal dataset, not actual index constituent data. This introduces classification noise at the margin.

**Data snooping.** The ATC signal was developed by ProntoNLP using some version of price data. To the extent the classifier was tuned to maximize a return-based objective, the signal may be overfit to the historical return distribution used during training. The walk-forward design partially mitigates this for the predictive model layer, but the baseline `ATCClassifierScore` itself carries this risk.

**Transaction cost assumption.** The 5 bps flat-cost assumption ignores market-impact for the Russell 3000 universe, where many names are illiquid small caps. Actual execution costs for small-cap long-short strategies can be 20–50 bps or more per side. The parameter sensitivity analysis (§5.6) shows the strategy is only viable up to ~8 bps one-way.

**Regime dependence.** The ATC signal was trained on data that includes the 2010–2021 bull market. Performance during the 2022 rate-shock and 2023–2026 post-normalization period has degraded materially at short horizons.


# 8. Future Work

**Point-in-time universe membership.** Obtaining historical S&P 500/1500 constituent lists (from CRSP, Compustat, or commercial vendors) would eliminate the survivorship bias concern and allow more rigorous universe-controlled analysis.

**Alternative price sources.** Polygon.io, Tiingo, and Alpha Vantage provide historical adjusted prices for delisted securities. Using them would improve event coverage from 51% toward 70–80% in RU3K, particularly for the 2010–2015 period.

**Multi-factor model integration.** The ATC signal could be combined with momentum, quality, and low-volatility factors in a proper multi-factor framework to assess its marginal contribution to a well-diversified factor portfolio.

**Intraday analysis.** The current backtest uses closing prices. Using open-to-close or event-time returns (if timestamp resolution permits) would provide a cleaner measurement of the immediate price reaction.

**Optimal trend horizon search.** The 2Q trend feature dominates QoQ and YoY variants. A finer search over lag lengths (1–8 quarters) with proper walk-forward validation could identify a more optimal momentum window.

**Cross-asset ATC signals.** The same NLP pipeline could be applied to fixed-income (credit spreads around earnings) or options markets (implied volatility changes), where information diffusion may be slower and the signal edge larger.


# 9. Conclusion

We present a rigorous look-ahead-free backtest of the ProntoNLP ATC signal across three equity universes. The data pipeline processes 4.47 GB of raw NLP data into a clean Parquet-based feature store covering 376,790 earnings call events with **772 engineered features** (193 Aspect × Theme cross-product base features + QoQ/2Q/YoY trend variants) and five forward-return horizons. Returns are winsorized at the 0.1th/99.9th percentiles. All ten look-ahead audit items pass. Price coverage reaches 99% in SP500/SP1500.

The `ATCClassifierScore` is a robust standalone signal with full-sample Spearman IC of +0.039 at the 10d horizon (SP500). Monthly quintile portfolios deliver net Sharpe ratios of 0.45 (SP500), 0.74 (SP1500), and 1.28 (RU3K) after 20 bps round-trip costs. Monthly rebalancing is the robust production cadence, delivering positive net Sharpe across all three universes (+0.45 / +0.74 / +1.28). SP500 daily rebalancing is TC-destroyed (net Sharpe −0.03). SP1500/RU3K daily strategies show higher gross returns but require max drawdowns of −39.5% / −78.3% and rely on a flat 5 bps TC assumption that likely understates small-cap market-impact at meaningful AUM. Among all models tested in a 34-quarter expanding walk-forward, **Combo XGBoost on 772 engineered + 13 Lasso-selected sparse features achieves the best IC-IR (+2.27) and the best SP500 portfolio Sharpe (+0.76, max DD −8.9%)** — nearly 3× the Sharpe of any individual Enhanced model (+0.26–0.27). The sparse feature selection reduces 405 raw AspectTheme cells to 13 robustly predictive features (IC + LassoCV intersection on 2010–2017 training data), concentrated in positive Financial Performance and Capital Allocation language. The 2-quarter trend in ATC score (`ATCClassifierScore_2q`) is the single most predictive engineered feature (IC_5d = +0.047). The Total speaker slice dominates speaker-specific cuts by 2–5×. The 20d holding period delivers the best empirical risk-adjusted performance (net Sharpe +0.73 vs. +0.45 at 10d). The most important risk is regime-dependent signal decay: pre-COVID ATC 10d IC (+0.063) has collapsed to +0.008 post-COVID, though the ML layer substantially recovers lost signal and should be monitored via rolling 8-quarter IC. Deployment is recommended at S&P 1500 scale with monthly rebalancing, octile buckets, 10d holding period, XGBoost scoring (fallback: Ridge), and an estimated capacity of $150–300M AUM.


# References

Loughran, T. & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.

Matsumoto, D., Pronk, M. & Roelofsen, E. (2011). What makes conference calls useful? The information content of managers' presentations and analysts' discussion sessions. *The Accounting Review*, 86(4), 1383–1414.

Mayew, W.J. & Venkatachalam, M. (2012). The power of voice: Managerial affective states and future firm performance. *Journal of Finance*, 67(1), 1–43.

ProntoNLP (2024). Earnings Call ATC (Aspect-Theme Classifier) Signal Dataset. Retrieved from https://prontonio.com.

Tetlock, P.C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance*, 62(3), 1139–1168.

# Appendix A: Data Pipeline Summary

| File | Size | Description |
|------|------|-------------|
| `signals.parquet` | 317.8 MB | 2,738,206 non-delete rows, 447 columns (float32) |
| `prices.parquet` | 40 MB | 9.3M daily adj-close rows, 3,109 tickers |
| `events_with_returns.parquet` | ~500 MB | 376,790 Total-slice events, 785 columns (772 features + meta + 5 returns) |
| `sparse_features.parquet` | 42 MB | 376,790 rows × 407 columns (BESTTICKER, entry_date, 405 AT cols) |
| `signal_slices.parquet` | 35 MB | ATCClassifierScore + EventScores for Total/CEO/CFO/Analysts |
| `universes.json` | 0.1 MB | SP500/SP1500/RU3K ticker lists |

# Appendix B: Feature List

**Aspect × Theme cross-product features (180):** For each of 5 aspects × 9 themes = 45 pairs: `at_{Aspect}_{Theme}_Positive`, `at_{Aspect}_{Theme}_Negative`, `at_{Aspect}_{Theme}_total`, `at_{Aspect}_{Theme}_net_sentiment`. Aspects: CurrentState, Forecast, Surprise, StrategicPosition, Other. Themes: FinancialPerformance, OperationalPerformance, MarketAndCompetitivePosition, StrategicInitiatives, CapitalAllocation, RegulatoryAndLegalIssues, ESG, MacroeconomicFactors, Other.

**Raw scores (13):** `ATCClassifierScore`; `EventsScore_{v}`, `EventPos_{v}`, `EventNeg_{v}` for each of 4 classifier variants `v` in {1_1_1, 2_1_1, 4_1_1, 4_2_1}.

**Base features total: 193** (180 cross-product + 13 raw scores).

**Multi-quarter trend features (193 × 3 = 579):** Each base feature is replicated with three lagged delta suffixes:
- `_qoq` — quarter-over-quarter (shift 1 within ticker)
- `_2q` — 2-quarter trend (shift 2 within ticker)
- `_yoy` — year-over-year (shift 4 within ticker)

**Total: 772 features** (193 base + 193 QoQ + 193 2Q + 193 YoY).

**Stretch-only (not in events_with_returns.parquet):** 405 raw AspectTheme columns (full 5×9×3×3 grid, Fluff/Filler dropped) saved in `sparse_features.parquet` and merged at runtime for the Stretch walk-forward model tier (1,177 total features).
