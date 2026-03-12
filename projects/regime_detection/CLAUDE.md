# CLAUDE.md — Regime Detection & Factor Risk Dashboard

## What this is
A Hidden Markov Model that detects 2-3 market regimes (Bull / Stress / Crisis)
from SPY returns, VIX, and realized vol, then shows how factor exposures,
correlations, and risk metrics shift by regime. Demonstrates decision-support
skills — not just measuring risk, but showing how risk changes across environments.

## Pipeline overview
1. `data/fetch_data.py` — Fetch SPY, VIX, FF daily factors via yfinance + Ken French library
2. `src/regime_model.py` — HMM fitting, walk-forward regime detection, label sorting
3. `src/factor_analysis.py` — Factor stats/correlations/exposures by regime
4. `src/risk_metrics.py` — Regime-conditional VaR, CVaR, vol, summary table
5. `src/charts.py` — Static matplotlib charts → results/
6. `run_pipeline.py` — End-to-end orchestration (8 steps)
7. `app.py` — Streamlit interactive dashboard (4 tabs)

## How to run
```bash
pip install hmmlearn seaborn  # new dependencies
cd projects/regime_detection
python run_pipeline.py        # first run downloads data (cached after)
python -m pytest tests/ -v    # ~38 tests, synthetic data only
streamlit run app.py           # interactive dashboard
```

## Key design decisions
- SPY + VIX + realized vol as HMM features (3 observable signals)
- VIX z-scored with expanding window to avoid lookahead bias
- Walk-forward with quarterly refit (same pattern as vol_forecasting GARCH)
- States sorted by mean return after each fit to handle label switching
- FF daily factors from Ken French library (not ETF proxies) for factor analysis
- 3-state model by default: Bull / Stress / Crisis (2-state also supported)

## Conventions
- `matplotlib.use("Agg")` at top of run_pipeline.py
- `plt.close(fig)` after charts, savefig to results/
- Cached data in `data/*.parquet` (gitignored)
- All vol outputs are annualized (multiply by sqrt(252))

## Dependencies
hmmlearn, seaborn (new), plus standard stack (pandas, numpy, matplotlib, etc.)

## Tests
`tests/test_regime.py` — ~38 tests covering all modules with synthetic data.
Run: `python -m pytest tests/ -v` from this directory.
