# CLAUDE.md — Volatility Forecasting Project

## What this is
A volatility forecasting comparison and vol-targeting backtest. Compares four
models (EWMA, GARCH, HAR-RV, VIX implied) on SPY daily data (2005-2024) and
implements the vol-targeting strategy from Paleologo Ch 6.

## Pipeline overview
1. `data/fetch_data.py` — Fetch SPY prices, VIX, 50-stock universe via yfinance, cache to parquet
2. `src/realized_vol.py` — Realized vol, forward RV target, HAR features, vol cone
3. `src/models.py` — EWMA (lambda=0.94), GARCH(1,1), HAR-RV (Corsi 2009), VIX implied
4. `src/evaluate.py` — QLIKE, MSE, MAE, Mincer-Zarnowitz regression, comparison charts
5. `src/vol_target.py` — Vol-targeting backtest (Paleologo Ch 6), performance metrics
6. `run_pipeline.py` — End-to-end orchestration (6 steps)

## How to run
```bash
pip install arch  # only new dependency
cd projects/vol_forecasting
python run_pipeline.py    # first run downloads data (cached after)
python -m pytest tests/ -v
```

## Data flow
- SPY daily adj close → daily returns → realized vol + HAR features
- Forward RV (22-day) = prediction target
- Each model produces annualized vol forecast series
- All forecasts aligned to common dates → evaluation
- 50-stock universe → equal-weight portfolio → vol-targeting backtest

## Key design decisions
- SPY only for vol forecasting (single asset, deep history)
- 50 liquid S&P stocks for vol-targeting (diversified portfolio)
- Start from 2005 to capture 2008 GFC
- GARCH refits every 22 days (daily refit too slow)
- HAR-RV uses expanding window OLS
- Vol target = 10% annualized (typical institutional target)
- Max leverage 2x, min leverage 0.1x (realistic constraints)

## Conventions
- `matplotlib.use("Agg")` at top of run_pipeline.py
- `plt.close(fig)` after charts, savefig to results/
- Cached data in `data/*.parquet` (gitignored)
- All vol outputs are annualized (multiply by sqrt(252))

## Tests
`tests/test_vol_models.py` — ~30 tests covering all modules with synthetic data.
Run: `python -m pytest tests/ -v` from this directory.
