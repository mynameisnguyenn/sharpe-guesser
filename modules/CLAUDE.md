# CLAUDE.md — Learning Modules

## What these are
Five progressive learning modules that teach quant finance concepts through Python.
Each module is a standalone script with explanations, computations, and exercises.

## Module progression
1. `module_1_statistics.py` — Descriptive stats, distributions, normality tests, correlation, stationarity
2. `module_2_risk_metrics.py` — VaR (3 methods), CVaR, drawdowns, Sortino, Calmar
3. `module_3_factor_models.py` — CAPM, Fama-French 3-factor, rolling beta, alpha, Information Ratio
4. `module_4_portfolio_optimisation.py` — Mean-variance, min variance, max Sharpe, risk parity, Black-Litterman
5. `module_5_strategies.py` — Momentum, pairs trading, SimpleBacktester, strategy comparison

## How to run
```bash
python -m modules.module_1_statistics
```
Must use `-m` flag because modules import from each other and from third-party packages.

## Conventions
- Each module has a `main()` function that runs a demo with real market data (via yfinance)
- All functions are self-contained — can be imported independently
- `plt.close()` after every chart (no interactive windows, no savefig)
- matplotlib Agg backend is NOT set here (it's set in sharpe_101.py which is a different entry point)
- TRADING_DAYS = 252 constant in each module
- Exercises printed at the end of each module's main()

## Dependencies between modules
- Module 2 previously imported from sharpe_101 — that was broken and inlined
- Module 3's `fetch_returns` is imported by `factor_dashboard.py`
- Module 5 is standalone (SimpleBacktester class)
- Modules 4 and 5 use `scipy.optimize` and `statsmodels`

## Tests
All module tests are in `tests/test_modules.py` (54 tests). Tests use synthetic data only.
