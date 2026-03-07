# CLAUDE.md — Tests

## What these test
Unit tests for the learning toolkit code (root-level scripts and modules/).
96 tests total, all using synthetic/deterministic data. No network calls.

## Test files
- `test_sharpe.py` (25 tests) — Tests for `sharpe_101.py` functions:
  simple_returns, log_returns, excess_returns, annualized_return/vol, sharpe_ratio, rolling_sharpe

- `test_modules.py` (54 tests) — Tests for modules 1-5:
  VaR, CVaR, drawdowns, Sortino, Calmar, CAPM regression, rolling beta, information ratio,
  portfolio optimization, risk parity, SimpleBacktester

- `test_factor_dashboard.py` (17 tests) — Tests for factor_dashboard.py:
  parse_args, analyse_ticker, print_comparison_table, save_comparison_chart

## How to run
```bash
python -m pytest tests/ -v
```

## Conventions
- All tests use `np.random.seed()` for deterministic results
- Mock `yfinance` downloads and `matplotlib.pyplot` where needed
- Use `pytest.approx()` for floating-point comparisons
- Test edge cases: empty data, short date ranges, all tickers failing
- Fixtures defined at top of each file
- No tests should make network calls

## Project tests
The Kelly ML project has its own tests at `projects/empirical_asset_pricing/tests/test_features.py`
(23 tests). Those test feature engineering functions with synthetic data.
