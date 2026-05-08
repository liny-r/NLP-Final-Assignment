"""
Build reports/backtest_charts.pdf — one chart per page, with title and caption.
Usage: python build_charts_pdf.py
"""
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

RESULTS = Path("reports") / "output"
OUTPUT  = Path("reports") / "backtest_charts.pdf"

# Ordered sections: (filename, section_label, caption)
FIGURES = [
    # ── Part 1: Baseline signal ───────────────────────────────────────────
    ("ic_annual_heatmap.png",
     "§1a  Annual IC Heatmap",
     "Spearman IC of ATCClassifierScore vs. forward returns (1–20d), by year and universe.\n"
     "Consistent positive IC across 2010–2026; temporary dip in 2020 (COVID regime change)."),

    ("ic_by_sector.png",
     "§1b  IC by GICS Sector",
     "Spearman IC by sector (SP500, SP1500, RU3K). Consumer Staples and Energy lead;\n"
     "Financials and Consumer Discretionary are weakest."),

    ("quintile_equity_curves.png",
     "§2a  Quintile Portfolio — Long / Short / L/S Equity Curves",
     "Monthly quintile L/S equity curves (10d holding, 5 bps one-way TC). Net Sharpe:\n"
     "SP500 +0.45, SP1500 +0.74, RU3K +1.28."),

    ("decile_drawdown_rolling_sharpe.png",
     "§2b  Decile Portfolio — Cumulative Return, Drawdown, Rolling Sharpe",
     "Top decile long / bottom decile short (D10−D1), monthly, 10d return, net of TC.\n"
     "Rolling 12-month Sharpe and drawdown shown per universe."),

    ("decile_spread_heatmap.png",
     "§2b  Decile Spread Heatmap (D10−D1 net, bps)",
     "Net decile spread by universe × horizon. Spread grows monotonically from 1d to 20d,\n"
     "consistent with the ATC classifier's 14-day training window."),

    ("alpha_decay.png",
     "§2c  Alpha Decay — IC vs. Return Horizon",
     "Spearman IC of ATCClassifierScore at 1, 3, 5, 10, 20-day horizons (all universes).\n"
     "IC peaks at 10–20d, supporting monthly rebalancing as the primary cadence."),

    ("cadence_comparison.png",
     "§2c  Cadence Comparison — Daily / Weekly / Monthly",
     "Quintile L/S cumulative equity curves across all cadences and universes.\n"
     "Monthly is the only cadence with positive net Sharpe for all three universes."),

    ("cadence_by_universe.png",
     "§2c  Primary Cadence: Monthly (all universes)",
     "Composite equity curves using Monthly rebalancing. SP500 daily is TC-destroyed\n"
     "(net Sharpe −0.03); SP1500/RU3K daily shows higher drawdown risk."),

    ("turnover_bar.png",
     "§2d  Turnover — Q5 Name-Level Churn",
     "Fraction of Q5 (long) names replaced each month. Near-100% turnover confirms\n"
     "the full-turnover TC assumption used throughout."),

    ("gross_net_exposure.png",
     "§2e  Gross / Net Exposure",
     "Equal-weight quintile construction produces an approximately dollar-neutral book.\n"
     "Net exposure remains close to zero across the full sample."),

    # ── Part 2: Engineered features & walk-forward ────────────────────────
    ("ic_engineered_features.png",
     "§1c  IC of Engineered Features (S&P 500)",
     "Spearman IC of key Aspect×Theme cross-product features vs. ATCClassifierScore baseline.\n"
     "CurrentState×FinPerf 2Q delta and ATCClassifierScore_2q are the strongest individual features."),

    ("ic_feature_horizon_heatmap.png",
     "§1d  Feature × Horizon IC Heatmap (Top 30 Features)",
     "Top 30 engineered features ranked by |IC| at 5d (SP500). ATCClassifierScore and its\n"
     "trend variants (2q, QoQ, YoY) dominate across all horizons."),

    ("walkforward_baseline_ic.png",
     "§2e  Walk-Forward Baseline IC (ATCClassifierScore, no model)",
     "Quarterly out-of-sample Spearman IC, 2018Q1–2026Q2. Mean IC +0.031 (All universes),\n"
     "IR +1.12. Visible post-COVID decay."),

    ("wf_portfolio_comparison.png",
     "§4b  Walk-Forward Portfolio — ATC Baseline vs. Ridge vs. LightGBM",
     "Monthly quintile L/S equity curves from OOS walk-forward predictions (all universes,\n"
     "20 bps round-trip TC). Ridge IR +1.96 outperforms LightGBM IR +1.64 on portfolio Sharpe."),

    ("wf_subperiod_ir.png",
     "§4d  Walk-Forward IC — Sub-Period Breakdown",
     "Information ratio by sub-period (Pre-COVID, COVID era, Post-COVID) for each model tier.\n"
     "ML models partially recover the signal decay visible in the raw ATC baseline."),

    ("feature_importance.png",
     "§4c  LightGBM Feature Importance (Last Walk-Forward Quarter)",
     "Top features by gain in the final walk-forward quarter. ATCClassifierScore trend variants\n"
     "and cross-product features dominate. Confirms 2q trend as the most valuable predictor."),

    # ── Part 3: Sparse matrix + combined model ────────────────────────────
    ("stretch_portfolio_comparison.png",
     "§3d  Portfolio Comparison — All Tiers (SP500, Monthly)",
     "OOS monthly quintile L/S equity curves: ATC Baseline, Enhanced Ridge/LGB,\n"
     "Combo Ridge/LGB, Combo XGBoost. XGBoost net Sharpe +0.76 dominates all others."),

    ("stretch_feature_importance.png",
     "§3  Combo XGBoost Feature Importance",
     "Top features in the Combo XGBoost model (772 engineered + 13 Lasso-selected sparse).\n"
     "Sparse AspectTheme cells concentrated in positive FinPerf and CapAlloc language."),

    # ── Robustness ────────────────────────────────────────────────────────
    ("ic_by_cap_bucket.png",
     "§5c  IC by Market-Cap Bucket",
     "Spearman IC of ATCClassifierScore stratified by market-cap quartile (universe proxy).\n"
     "Signal is consistent across size buckets — not exclusively a small-cap effect."),

    ("parameter_sensitivity.png",
     "§5.6  Parameter Sensitivity — TC Level / Bucket Count / Return Horizon",
     "(A) Break-even near 8 bps one-way. (B) Octile (8 buckets) is empirically optimal.\n"
     "(C) 20d holding period maximises net Sharpe (+0.73 vs +0.45 at 10d)."),

    ("walkforward_ic.png",
     "§5.3  Walk-Forward IC Summary — All Model Tiers",
     "Quarterly OOS IC for all 7 model variants (2018Q1–2026Q2). Combo XGBoost IR +2.27\n"
     "is the highest; ATC baseline IR +1.12 serves as the benchmark floor."),
]

def make_charts_pdf(figures, output_path):
    with PdfPages(output_path) as pdf:
        # ── Title page ───────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.65, "Backtesting the ProntoNLP Earnings-Call ATC Signal",
                ha="center", va="center", fontsize=20, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.55, "Backtest Charts Bundle",
                ha="center", va="center", fontsize=14, color="#444",
                transform=ax.transAxes)
        ax.text(0.5, 0.45, "Yueqi Lin  ·  May 2026",
                ha="center", va="center", fontsize=12, color="#666",
                transform=ax.transAxes)
        ax.text(0.5, 0.32,
                f"{len(figures)} figures  ·  S&P 500 / S&P 1500 / Russell 3000\n"
                "5 bps one-way TC  ·  Expanding walk-forward 2018Q1–2026Q2",
                ha="center", va="center", fontsize=10, color="#888",
                transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── One figure per page ──────────────────────────────────────────
        for fname, section, caption in figures:
            img_path = RESULTS / fname
            if not img_path.exists():
                print(f"  SKIP (missing): {fname}")
                continue

            fig = plt.figure(figsize=(11, 8.5))

            # Section label at top
            fig.text(0.05, 0.97, section, fontsize=11, fontweight="bold",
                     color="#0B2545", va="top")
            # Caption at bottom
            fig.text(0.05, 0.02, caption, fontsize=8, color="#444",
                     va="bottom", wrap=True)

            # Image occupies the middle band
            ax_img = fig.add_axes([0.03, 0.09, 0.94, 0.86])
            ax_img.axis("off")
            img = mpimg.imread(img_path)
            ax_img.imshow(img, aspect="equal")

            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  Added: {fname}")

    print(f"\nSaved: {output_path}  ({output_path.stat().st_size / 1024:.0f} KB)")

if __name__ == "__main__":
    print(f"Building {OUTPUT} ...")
    make_charts_pdf(FIGURES, OUTPUT)
