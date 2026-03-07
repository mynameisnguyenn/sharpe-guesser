# CLAUDE.md — Empirical Asset Pricing via Machine Learning

## What this is
A simplified replication of Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine
Learning." This is a PORTFOLIO SHOWCASE piece — it should demonstrate real quant research
skills to hiring managers.

## Pipeline overview
1. `data/fetch_data.py` — Scrape S&P 500 tickers from Wikipedia, download daily OHLCV via
   yfinance, compute monthly returns, cache to parquet
2. `src/features.py` — 7 features: mom_1m, mom_6m, mom_12m, realized_vol, log_dollar_vol,
   volume_trend, max_return. Daily features, resampled to monthly in the pipeline.
3. `src/models.py` — ElasticNetCV and RandomForestRegressor with `expanding_window_predict`
   (walk-forward, no lookahead bias). min_periods=36 months.
4. `src/portfolio.py` — Rank into deciles, long-short spread, performance metrics
5. `src/evaluate.py` — Strategy comparison, feature importance, cumulative return plots,
   decile return bar charts
6. `run_pipeline.py` — End-to-end orchestration script

## How to run
```bash
cd projects/empirical_asset_pricing
python run_pipeline.py    # first run downloads ~500 stocks (cached after)
```

## Data flow
- Daily prices/volumes (wide: dates x tickers) → features (long: date,ticker MultiIndex)
- Features resampled to monthly (last daily value per month per ticker)
- Monthly returns stacked to long format, merged with features
- Target: `next_month_return` = `groupby("ticker")["monthly_return"].shift(-1)`
- expanding_window_predict walks forward month by month

## Key design decisions
- S&P 500 only (not full CRSP) — survivorship bias acknowledged in README
- 7 features (not 94) — focused on what the paper shows matters most
- Two models (elastic net, random forest) — not neural networks
- Expanding window (not rolling) — matches the paper
- Decile sorts for evaluation — standard academic methodology

## Conventions
- `matplotlib.use("Agg")` at top of run_pipeline.py
- `plt.close(fig)` after charts, savefig to results/ directory
- Cached data in `data/*.parquet` (gitignored)
- Results (PNGs) in `results/` (gitignored via *.png in root .gitignore — may want to track these)

## Latest results (2026-03-07)
- Elastic Net L/S: Sharpe 0.12, annual return 7.2%, max DD -35.7%
- Random Forest L/S: Sharpe 0.11, annual return 6.8%, max DD -26.3%
- Decile monotonicity present in both models (D10 ~24% vs D1 ~17% annualized)
- 501 stocks downloaded, 2 failed (Q, SNDK — delisted)

## Tests
`tests/test_features.py` — 23 tests for feature functions with synthetic data.
Run: `python -m pytest tests/ -v` from this directory.
