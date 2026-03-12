# Regime Detection & Factor Risk Dashboard

A Hidden Markov Model that detects market regimes (Bull / Stress / Crisis) from SPY returns, VIX, and realized volatility, then quantifies how factor exposures, correlations, and risk metrics shift across environments.

## Why This Matters

Standard risk metrics (VaR, CVaR, volatility) blend calm and crisis periods into a single number. This understates tail risk — a portfolio manager who only sees unconditional 95% VaR doesn't realize that crisis-regime VaR might be 2-3x worse. Regime detection answers: **"what mode is the market in right now, and how does that change our risk?"**

## Methodology

### Hidden Markov Model
- **Observable features:** SPY daily returns, VIX z-score (expanding window, no lookahead), 22-day realized vol
- **States:** 2 or 3 latent regimes, sorted by mean return after fitting (handles label switching)
- **Walk-forward estimation:** Expanding window with quarterly refit — same pattern used for GARCH in production risk systems
- **Library:** `hmmlearn` (Gaussian HMM with full covariance)

### Factor Analysis
- **Fama-French factors:** MKT-RF, SMB, HML from Ken French's daily data library
- **Per-regime statistics:** Annualized mean, vol, Sharpe, skewness
- **Correlation breakdown:** Shows how factor correlations spike in crises (diversification fails when you need it)
- **Regime-conditional betas:** OLS regression of SPY on factors, separately per regime

### Risk Metrics
- **Regime-conditional VaR/CVaR:** Historical percentile method, per regime vs unconditional
- **Risk underestimation ratio:** How much worse crisis-regime risk is vs the blended number
- **Regime summary:** Days, proportion, return, vol, Sharpe, max drawdown per regime

## Project Structure

```
regime_detection/
├── data/
│   └── fetch_data.py          # SPY, VIX, FF daily factors → parquet cache
├── src/
│   ├── regime_model.py        # HMM fitting, walk-forward regime detection
│   ├── factor_analysis.py     # Factor stats/correlations by regime
│   ├── risk_metrics.py        # Regime-conditional VaR, CVaR, vol
│   └── charts.py              # Static matplotlib charts → results/
├── tests/
│   └── test_regime.py         # 41 tests, synthetic data only
├── results/                   # PNG charts (tracked in git)
├── run_pipeline.py            # 8-step end-to-end pipeline
├── app.py                     # Streamlit interactive dashboard (4 tabs)
├── requirements.txt
├── CLAUDE.md
└── RESULTS.md
```

## How to Run

```bash
# Install dependencies
pip install hmmlearn seaborn

# Run the pipeline (generates charts in results/)
cd projects/regime_detection
python run_pipeline.py

# Run tests
python -m pytest tests/ -v

# Launch interactive dashboard
streamlit run app.py
```

First run downloads data from Yahoo Finance and Ken French's library (~30 seconds). Subsequent runs use the parquet cache.

## Dashboard (4 Tabs)

1. **Regime Overview** — Current regime indicator, SPY price with regime-colored background, summary table
2. **Factor Behavior** — Factor stats table by regime, Sharpe bar chart, correlation heatmaps per regime
3. **Risk Metrics** — Unconditional vs regime-conditional VaR/CVaR table, vol bars, return distributions
4. **Transitions** — Transition matrix heatmap, expected durations, regime frequency by year

## References

- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*.
- Ang, A. and Bekaert, G. (2002). "International Asset Allocation with Regime Shifts." *Review of Financial Studies*.
- Paleologo, G. (2021). *Advanced Portfolio Management*. Wiley.
- Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics*.

## FAQ

**Q: Why HMM instead of k-means or threshold rules?**
HMMs model the sequential structure of markets — regimes persist and transition gradually. K-means treats each day independently, ignoring that yesterday's regime informs today's. Threshold rules (e.g., "VIX > 30 = crisis") are arbitrary and brittle.

**Q: Why 3 states?**
Empirically, 3 states capture the common pattern: normal markets (Bull), elevated uncertainty (Stress), and crisis (Crisis). The dashboard supports 2 states as well. More states risk overfitting with daily data.

**Q: How does this avoid lookahead bias?**
VIX is z-scored with an expanding window (only past data). Walk-forward regime detection refits the HMM quarterly using only data available at that point. No future information leaks into regime labels.

**Q: Why do crisis correlations matter?**
In normal markets, assets have moderate correlations and diversification works. In crises, correlations spike toward 1 — everything falls together. A risk model that uses unconditional correlations will overstate diversification benefits precisely when they disappear.
