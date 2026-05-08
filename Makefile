## Backtesting the ProntoNLP Earnings-Call ATC Signal
## Usage: make all          → run full pipeline from CSV to PDF
##        make data         → data prep only (00_data_prep.ipynb)
##        make analysis     → analysis notebook (01_analysis.ipynb)
##        make tests        → look-ahead bias tests (02_lookahead_tests.ipynb)
##        make report       → generate PDF from markdown
##        make clean        → remove generated Parquet / PNG files

PYTHON        = python
JUPYTER_FLAGS = --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200
PDF_ENGINE    = xelatex
PANDOC_FLAGS  = --pdf-engine=$(PDF_ENGINE) -V geometry:margin=1in -V fontsize=11pt \
                -V "mainfont=STIX Two Text" -V "mathfont=STIX Two Math"

.PHONY: all data analysis tests report clean

all: report

## ── Step 1: data pipeline ─────────────────────────────────────────────────
data/events_with_returns.parquet: 00_data_prep.ipynb
	jupyter nbconvert $(JUPYTER_FLAGS) 00_data_prep.ipynb

data: data/events_with_returns.parquet

## ── Step 2: analysis (IC, portfolios, walk-forward, robustness) ───────────
results/walkforward_ic.png: 01_analysis.ipynb data/events_with_returns.parquet
	jupyter nbconvert $(JUPYTER_FLAGS) 01_analysis.ipynb

analysis: results/walkforward_ic.png

## ── Step 3: formal look-ahead bias tests ─────────────────────────────────
reports/look_ahead_audit.md: 02_lookahead_tests.ipynb results/walkforward_ic.png
	jupyter nbconvert $(JUPYTER_FLAGS) 02_lookahead_tests.ipynb

tests: reports/look_ahead_audit.md

## ── Step 4: PDF report ────────────────────────────────────────────────────
reports/research_report.pdf: reports/research_report.md results/walkforward_ic.png
	cd reports && pandoc research_report.md \
		-o research_report.pdf \
		$(PANDOC_FLAGS)

report: reports/research_report.pdf

## ── Clean generated artefacts (keeps raw CSV and Parquet inputs) ──────────
clean:
	rm -f data/signals.parquet data/prices.parquet \
	      data/events_with_returns.parquet data/signal_slices.parquet
	rm -f results/*.png
	find . -name "*.pyc" -delete
