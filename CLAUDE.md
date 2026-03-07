# CLAUDE.md — Sharpe Guesser Project Root

## What this repo is
A quant finance learning toolkit AND portfolio showcase for a risk & data associate
transitioning from a fundamental hedge fund to quantitative roles. Two tracks:

1. **Learning toolkit** (root): `sharpe_101.py`, `sharpe_guesser.py`, `factor_dashboard.py`,
   `modules/`, `notebooks/` — teaches Python through finance concepts
2. **Portfolio projects** (`projects/`): self-contained showcase pieces that demonstrate
   real quant skills to hiring managers

## User background
- Risk and data associate at a fundamental tiger-cub hedge fund
- Learning to code from scratch — does NOT currently program
- Reads risk reports daily (VaR, beta, factor exposures) — knows the concepts, learning the implementation
- Reading Paleologo's "Advanced Portfolio Management"
- Target roles: quant risk, quant research, systematic strategies

## Coding conventions
- Python 3.13, pandas-first approach
- Finance examples always (no foo/bar)
- Readability over performance unless 1M+ rows
- Brief inline comments on non-obvious lines only
- One-line docstrings unless complex
- matplotlib Agg backend for scripts (no interactive chart windows)
- `plt.close()` after charts, never `plt.show()` in scripts
- No `plt.savefig()` in learning modules (clutters local folder)
- `plt.savefig()` IS used in projects (saves to results/ directories)
- yfinance for market data
- pytest with synthetic data for tests (no network calls)

## How to run things
```bash
# Activate venv first
.\venv\Scripts\Activate.ps1   # PowerShell
source venv/Scripts/activate   # Git Bash

# Learning scripts
python sharpe_101.py
python sharpe_guesser.py --rounds 5
python factor_dashboard.py AAPL MSFT JPM

# Learning modules
python -m modules.module_1_statistics
python -m modules.module_2_risk_metrics

# Tests
python -m pytest tests/ -v

# Projects (each has its own instructions)
cd projects/risk_dashboard && streamlit run app.py
cd projects/empirical_asset_pricing && python run_pipeline.py
```

## Key files
- `requirements.txt` — base dependencies (no scikit-learn, that's in project requirements)
- `tests/` — 96 tests covering sharpe_101, all 5 modules, and factor_dashboard
- `.gitignore` — excludes venv, __pycache__, *.png, parquet data, IDE files

## Known fixed bugs (for reference)
- `sharpe_ratio` zero-vol guard: `< 1e-10` not `== 0`
- `sortino_ratio` NaN guard: checks `np.isnan(dd)` in addition to `== 0`
- Module 2 broken import: inlined `sharpe_ratio` instead of importing from sharpe_101
- Wikipedia scraping: added User-Agent header to avoid 403
