# Quantitative Finance — Learning Toolkit & Portfolio Projects

A quant finance repo with two tracks: a **learning toolkit** that teaches Python through finance concepts, and **portfolio projects** that demonstrate real quant research skills.

## Portfolio Projects

### [Empirical Asset Pricing via Machine Learning](projects/empirical_asset_pricing/)
A simplified replication of [Gu, Kelly, Xiu (2020)](https://doi.org/10.1093/rfs/hhaa009) — the landmark paper on ML-driven cross-sectional return prediction.

- **3 models**: Elastic net, random forest, gradient boosted trees
- **28 features**: 7 base characteristics (momentum, volatility, size, liquidity) + 21 pairwise interactions
- **Walk-forward prediction**: Expanding window with no lookahead bias — the most important detail in backtesting
- **Full evaluation**: OOS R², Fama-French 3-factor alpha regression, t-test on L/S spread, turnover & transaction cost analysis
- **Decile portfolio sorts**: Long-short spread from prediction-ranked deciles (standard academic methodology)

```bash
cd projects/empirical_asset_pricing
python run_pipeline.py
```

### [Interactive Risk Dashboard](projects/risk_dashboard/)
A Streamlit dashboard that computes risk metrics, factor exposures, and comparative analysis for user-selected tickers.

- **3 tabs**: Risk Overview (Sharpe/Sortino/Calmar, VaR/CVaR, drawdown charts), Factor Exposure (CAPM, Fama-French 3-factor, rolling beta), Comparison (side-by-side metrics, cumulative returns, correlation heatmap)
- **Interactive controls**: Ticker input, date range, risk-free rate, rolling window, VaR confidence level
- **Plotly charts** with dual-axis overlays and educational expanders

```bash
cd projects/risk_dashboard
pip install -r requirements.txt
streamlit run app.py
```

### [Volatility Forecasting & Vol-Targeting](projects/vol_forecasting/)
Compares four vol forecast models on SPY (2005-2024) and implements the vol-targeting strategy from Paleologo Ch 6.

- **4 models**: EWMA (RiskMetrics), GARCH(1,1), HAR-RV (Corsi 2009), VIX implied
- **Proper evaluation**: QLIKE loss function, Mincer-Zarnowitz unbiasedness regression, MZ R² of 45-51%
- **Vol-targeting backtest**: Scales position by forecast vol → max drawdown drops from -43% to -16%, kurtosis from 12.3 to 2.4
- **40 tests**: All synthetic data, no network calls

```bash
pip install arch
cd projects/vol_forecasting
python run_pipeline.py
```

## Learning Toolkit

| File / Directory | What it teaches |
|------|---------|
| `sharpe_101.py` | Sharpe ratio from first principles — fetching data, computing returns, annualizing, rolling analysis |
| `sharpe_guesser.py` | Interactive game: guess the Sharpe ratio from a return chart (trains PM-level intuition) |
| `factor_dashboard.py` | CLI factor exposure dashboard — CAPM, Fama-French, rolling beta, Information Ratio |
| `modules/module_1_statistics.py` | Descriptive stats, distributions, normality tests, correlation, stationarity |
| `modules/module_2_risk_metrics.py` | VaR (historical, parametric, Monte Carlo), CVaR, drawdowns, Sortino, Calmar |
| `modules/module_3_factor_models.py` | CAPM regression, Fama-French 3-factor, rolling beta, alpha, Information Ratio |
| `modules/module_4_portfolio_optimisation.py` | Mean-variance, min variance, max Sharpe, risk parity, Black-Litterman |
| `modules/module_5_strategies.py` | Momentum, pairs trading, SimpleBacktester, strategy comparison |
| `notebooks/factor_math_interactive.ipynb` | Interactive factor math — tweak inputs, see charts update |
| `notebooks/factor_models_explained.ipynb` | Factor models walkthrough with real market data |

## Quick start

```bash
pip install -r requirements.txt

# Learn the concepts
python sharpe_101.py

# Test your intuition
python sharpe_guesser.py --rounds 5

# Run the factor dashboard
python factor_dashboard.py AAPL MSFT JPM

# Run tests (96+)
python -m pytest tests/ -v
```

## Who this is for

Risk and data professionals who want to learn programming through finance concepts they already understand — and build portfolio pieces that demonstrate real quant skills.

## References

- Gu, Kelly, Xiu (2020) — "Empirical Asset Pricing via Machine Learning", *Review of Financial Studies*
- Corsi (2009) — "A Simple Approximate Long-Memory Model of Realized Volatility", *Journal of Financial Econometrics*
- Paleologo (2021) — *Advanced Portfolio Management*
- Grinold & Kahn (2000) — *Active Portfolio Management*
