## Backtesting the ProntoNLP Earnings-Call ATC Signal
## Usage: make all          → run full pipeline from CSV to PDF
##        make data         → data prep only (00_data_prep.ipynb)
##        make analysis     → analysis notebook (01_analysis.ipynb)
##        make tests        → look-ahead bias tests (02_lookahead_tests.ipynb)
##        make report       → generate PDF from markdown
##        make charts       → bundle PNGs into backtest_charts.pdf
##        make wrds         → optional: WRDS / CRSP pipeline for PIT RU3K and §8a
##        make clean        → remove generated Parquet / PNG files

PYTHON        = python
JUPYTER_FLAGS = --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200
PDF_ENGINE    = xelatex
PANDOC_FLAGS  = --pdf-engine=$(PDF_ENGINE) -V geometry:margin=1in -V fontsize=11pt \
                -V "mainfont=STIX Two Text" -V "mathfont=STIX Two Math"

.PHONY: all data analysis tests report charts wrds clean

all: data analysis tests report charts

## ── Step 1: data pipeline ─────────────────────────────────────────────────
data/events_with_returns.parquet: 00_data_prep.ipynb
	jupyter nbconvert $(JUPYTER_FLAGS) 00_data_prep.ipynb

data: data/events_with_returns.parquet

## ── Step 2: analysis (IC, portfolios, walk-forward, robustness) ───────────
reports/output/walkforward_ic.png: 01_analysis.ipynb data/events_with_returns.parquet
	jupyter nbconvert $(JUPYTER_FLAGS) 01_analysis.ipynb

analysis: reports/output/walkforward_ic.png

## ── Step 3: formal look-ahead bias tests ─────────────────────────────────
reports/look_ahead_audit.md: 02_lookahead_tests.ipynb reports/output/walkforward_ic.png
	jupyter nbconvert $(JUPYTER_FLAGS) 02_lookahead_tests.ipynb

tests: reports/look_ahead_audit.md

## ── Step 4: PDF report ────────────────────────────────────────────────────
reports/research_report.pdf: reports/research_report.md reports/look_ahead_audit.md reports/output/walkforward_ic.png
	cd reports && pandoc research_report.md \
		-o research_report.pdf \
		$(PANDOC_FLAGS)

report: reports/research_report.pdf

## ── Step 5: backtest charts PDF ───────────────────────────────────────────
reports/backtest_charts.pdf: build_charts_pdf.py reports/output/walkforward_ic.png
	$(PYTHON) build_charts_pdf.py

charts: reports/backtest_charts.pdf

## ── Optional: WRDS / CRSP pipeline (requires WRDS credentials) ───────────
## Pulls survivorship-free CRSP daily prices and the PIT Russell 3000 proxy,
## merges them into events_with_returns_wrds.parquet, then runs the §8a
## yfinance-vs-CRSP comparison and the T9–T14 look-ahead tests.
## The price pull (~14.9 M rows) takes ~10–30 minutes the first time.
data/wrds_prices.parquet: 03_wrds_pull.py
	$(PYTHON) 03_wrds_pull.py

data/events_with_returns_wrds.parquet: 04_wrds_integrate.py \
                                       data/wrds_prices.parquet \
                                       data/events_with_returns.parquet
	$(PYTHON) 04_wrds_integrate.py

wrds: data/events_with_returns_wrds.parquet
	$(PYTHON) 05_wrds_compare.py
	$(PYTHON) 06_wrds_lookahead_tests.py

## ── Clean generated artefacts (keeps raw CSV and Parquet inputs) ──────────
clean:
	rm -f data/signals.parquet data/prices.parquet \
	      data/events_with_returns.parquet data/signal_slices.parquet \
	      data/sparse_features.parquet
	rm -f data/wrds_prices.parquet data/events_with_returns_wrds.parquet
	rm -rf data/wrds_prices_chunks
	rm -f reports/output/*.png
	find . -name "*.pyc" -delete
