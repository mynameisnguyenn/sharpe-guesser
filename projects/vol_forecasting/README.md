# Volatility Forecasting & Vol-Targeting

**A comparison of four volatility forecasting models on SPY (2005-2024) with a vol-targeting backtest implementing Paleologo Ch 6.**

---

## Overview

Volatility is the one thing in finance that's actually predictable. Unlike returns (where even 0.5% out-of-sample R-squared is considered good), realized vol is highly persistent and forecastable — models routinely achieve 40-50% R-squared. This makes vol forecasting a core skill for any quant risk role.

This project compares four canonical vol forecasting approaches on SPY daily returns over 2005-2024 (capturing the 2008 GFC, 2020 COVID crash, and the 2022 rate hike regime). It then uses the best forecast to implement a vol-targeting strategy from Paleologo's *Advanced Portfolio Management* (Chapter 6), demonstrating how vol forecasts translate into portfolio decisions.

## Project Structure

```
vol_forecasting/
├── data/
│   ├── __init__.py
│   └── fetch_data.py              # SPY, VIX, 50-stock universe → parquet cache
├── src/
│   ├── __init__.py
│   ├── realized_vol.py            # RV computation, forward RV, HAR features, vol cone
│   ├── models.py                  # EWMA, GARCH(1,1), HAR-RV, VIX implied
│   ├── evaluate.py                # QLIKE, MSE, MAE, Mincer-Zarnowitz, charts
│   └── vol_target.py              # Vol-targeting backtest (Paleologo Ch 6)
├── tests/
│   ├── __init__.py
│   └── test_vol_models.py         # 40 tests, all synthetic data
├── results/                       # Output charts
├── run_pipeline.py                # End-to-end orchestration (run this)
├── requirements.txt
├── CLAUDE.md
├── RESULTS.md
└── README.md
```

## How to Run

```bash
# Install the one new dependency
pip install arch

# Run the full pipeline
cd projects/vol_forecasting
python run_pipeline.py

# Run tests (40 tests, synthetic data, no network)
python -m pytest tests/ -v
```

First run downloads SPY (2005-2024), VIX, and 50 stocks via yfinance. Data is cached to parquet — subsequent runs skip the download.

## Models

### 1. EWMA (RiskMetrics) — The Industry Baseline

```
sigma²_t = lambda * sigma²_{t-1} + (1 - lambda) * r²_{t-1}
```

The simplest model: an exponentially weighted average of past squared returns. Lambda = 0.94 is the RiskMetrics standard (effectively a ~60-day half-life). It's what you'd see in a basic VaR model at most banks. Pure pandas implementation — no fitting required.

**Strengths:** Simple, fast, no estimation risk.
**Weaknesses:** No mean reversion — after a vol spike, EWMA stays elevated until it slowly decays. It can't capture the fact that vol tends to revert to a long-run average.

### 2. GARCH(1,1) — The Academic Workhorse

```
sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}
```

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) captures two things EWMA misses: (1) mean reversion — vol reverts to omega/(1-alpha-beta) over time, and (2) fitted parameters — alpha and beta are estimated from data, not fixed.

Uses an expanding window with refitting every 22 trading days. Between refits, applies the recursive formula with last fitted parameters. The `arch` package handles maximum likelihood estimation.

**Strengths:** Mean reversion, well-understood statistical properties.
**Weaknesses:** Assumes symmetric response to positive and negative returns (addressed by GJR-GARCH extension). Fitting can be slow.

### 3. HAR-RV (Corsi 2009) — The Multi-Horizon Model

```
RV_forward_22d = a + b1 * RV_daily + b2 * RV_weekly + b3 * RV_monthly
```

The Heterogeneous Autoregressive model decomposes vol into three frequencies corresponding to different types of market participants:
- **Daily RV (rv_d):** Day-trader horizon. Captures single-day shocks.
- **Weekly RV (rv_w):** Swing-trader horizon. 5-day rolling window.
- **Monthly RV (rv_m):** Macro/institutional horizon. 22-day rolling window.

The idea: volatility cascades from longer to shorter horizons. Monthly vol informs weekly vol, which informs daily vol. The OLS regression captures these relationships with expanding-window estimation.

**Strengths:** Captures multi-horizon structure, very strong empirical performance, simple to implement.
**Weaknesses:** Linear model — can't capture nonlinear vol dynamics (addressed by neural HAR extensions).

### 4. VIX Implied — The Market's Own Forecast

VIX is the CBOE Volatility Index — the market's risk-neutral expectation of 30-day S&P 500 vol. No model to fit: just divide by 100 (VIX = 20 means 20% annualized vol). This is the benchmark every model-based forecast needs to beat — if the market already prices in everything your model knows, your model adds no value.

**Strengths:** Incorporates all market information (forward-looking), no estimation risk, no lookahead.
**Weaknesses:** Contains a volatility risk premium (VIX systematically overstates realized vol), which means it's biased upward.

## Evaluation Framework

### Loss Functions

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **QLIKE** | mean(RV/forecast - log(RV/forecast) - 1) | Quasi-likelihood loss. Standard in vol forecasting. Penalizes under-prediction of high vol (exactly what risk managers want). |
| **MSE** | mean((RV - forecast)²) | Symmetric error. Can be dominated by outlier vol events. |
| **MAE** | mean(\|RV - forecast\|) | Robust to outliers. Gives equal weight to all errors. |

**Why QLIKE matters more than MSE:** QLIKE is the "right" loss function for vol because it penalizes relative errors, not absolute errors. Under-predicting a 40% vol spike as 20% is a much worse mistake than over-predicting 15% vol as 20% — QLIKE captures this asymmetry, MSE doesn't.

### Mincer-Zarnowitz Regression

```
RV_actual = alpha + beta * forecast + epsilon
```

The standard unbiasedness test. A perfect forecast has alpha = 0 and beta = 1. The R-squared tells you what fraction of realized vol variation your forecast explains.

- **Alpha > 0:** Your model systematically under-predicts vol.
- **Beta < 1:** Your model over-reacts to its own signal.
- **R² of 40-50%:** Typical for daily vol forecasts — vol is predictable but not perfectly so.

## Vol-Targeting Backtest (Paleologo Ch 6)

### The Idea

Instead of holding a fixed portfolio, scale your position size so that your **predicted dollar volatility stays constant**:

```
leverage_t = target_vol / forecast_vol_t
managed_return_t = leverage_t * raw_return_t
```

When vol is high (2008, March 2020), hold less. When vol is low, hold more. This is exactly what institutional risk managers do — they set vol budgets and scale positions accordingly.

### Why It Works

Paleologo's argument (Ch 6): high-vol periods tend to have negative expected returns due to the "leverage effect" (vol spikes when markets fall). By scaling down in high-vol regimes, you avoid the worst drawdowns while maintaining exposure in calm periods. The result:

- **Sharpe improves** (modestly, +0.03 to +0.10) because you avoid the worst risk-adjusted periods
- **Max drawdown drops dramatically** because you scale down before/during crashes
- **Return distribution normalizes** (kurtosis drops) — fewer extreme moves
- **Annualized return drops** because you're holding less on average — you trade return for risk reduction

### Implementation Details

- **Universe:** 50 liquid S&P 500 stocks (equal-weight)
- **Vol forecast:** EWMA on the portfolio itself (not SPY)
- **Target vol:** 10% annualized (typical institutional equity target)
- **Leverage bounds:** [0.1x, 2.0x] — never go below 10% invested or above 200%
- **Lag:** Leverage uses today's forecast to size TOMORROW's position (no lookahead)

## Data

All free via yfinance:

| Data | Ticker | Period | Purpose |
|------|--------|--------|---------|
| SPY daily prices | SPY | 2005-2024 | Primary forecast target |
| VIX daily close | ^VIX | 2005-2024 | Implied vol benchmark |
| 50 liquid stocks | See LIQUID_50 list | 2005-2024 | Vol-targeting universe |

Start date is 2005 to capture the 2008 GFC — the most important vol event in the sample and a critical stress test for any vol model.

## Frequently Asked Questions

### "Why does VIX have the highest MZ R-squared but not the lowest QLIKE?"

VIX has the highest R-squared (50.6%) because it's the most informative signal — it incorporates options market information that historical models can't access. But it doesn't have the lowest QLIKE because VIX systematically **overstates** realized vol (the "volatility risk premium"). Investors are willing to pay more for downside protection than fair value, which pushes VIX above expected realized vol by ~2-4 percentage points on average.

QLIKE penalizes this persistent over-prediction, so HAR-RV and GARCH (which don't have this bias) score better on QLIKE despite having lower R-squared.

### "What is the volatility risk premium?"

The VRP is the difference between implied vol (VIX) and subsequent realized vol. On average, VIX is ~2-4% higher than what actually materializes. This exists because investors are risk-averse and willing to overpay for portfolio insurance. Selling this premium (e.g., writing options) is one of the oldest systematic strategies in finance.

### "Why does vol-targeting reduce return but improve Sharpe?"

The vol-targeted portfolio has a lower average leverage (~0.5x in calm periods, because the unmanaged portfolio typically runs at ~19% vol against a 10% target). Lower leverage = lower expected return. But the Sharpe ratio divides return by vol, and since vol drops even more than return (from 19% to 10%), the ratio improves.

The real value is in the **tail**: max drawdown drops from -43% to -16%, and kurtosis drops from 12.3 to 2.4. The return distribution becomes much more normal and predictable.

### "Why not use GARCH or HAR for vol-targeting instead of EWMA?"

For this backtest, we use EWMA on the **portfolio** itself (not SPY), because:
1. EWMA is available from day 22 onward (GARCH/HAR need 252+ days of history)
2. We need vol of the PORTFOLIO, not vol of SPY — the 50-stock EW portfolio has different vol dynamics
3. For vol-targeting, any reasonable vol estimate works — the improvement comes from the strategy logic, not the precision of the forecast

In production, you'd typically use a more sophisticated model, but EWMA is the standard starting point.

### "What does GARCH capture that EWMA misses?"

Mean reversion. After a vol spike (like March 2020), EWMA slowly decays the elevated estimate over time. GARCH pulls vol back toward its long-run average more quickly because it has an explicit "unconditional variance" term (omega/(1-alpha-beta)). In practice, this means GARCH is faster to normalize after crises.

### "What is the HAR model's key insight?"

Different market participants operate at different horizons. A day-trader's reaction to a vol shock is immediate (daily RV). A swing trader's response plays out over a week (weekly RV). A pension fund's rebalancing happens over months (monthly RV). The HAR model captures this "heterogeneous" structure by regressing on all three horizons simultaneously. The result: it captures the multi-scale persistence of vol better than models that only look at one horizon.

### "How do I interpret the vol cone chart?"

The vol cone shows the historical distribution of realized vol at different horizons (5-day, 10-day, 22-day, etc.). The shaded bands show the 10th-90th and 25th-75th percentile ranges. The red dots show where CURRENT vol sits.

- Dots near the top of the cone → current vol is historically high (risk-off)
- Dots near the bottom → current vol is historically low (complacency)
- The cone typically widens for shorter horizons because short-term vol is noisier

### "What's the difference between realized vol and implied vol?"

- **Realized vol:** Backward-looking. How much the asset actually moved over the past N days. Computed from historical returns.
- **Implied vol:** Forward-looking. How much the options market expects the asset to move over the next N days. Extracted from option prices (Black-Scholes).

VIX is implied vol. Our forecast target (forward RV) is realized vol. The gap between them is the volatility risk premium.

## Tests

```bash
python -m pytest tests/ -v    # 40 tests, all synthetic data, no network calls
```

Tests cover:
- Realized vol computation and annualization
- Forward RV target alignment
- HAR feature properties (daily smoother than weekly)
- Vol cone percentile ordering
- EWMA lambda sensitivity
- GARCH expanding window and NaN handling
- HAR-RV non-negativity
- VIX conversion
- All loss functions (QLIKE, MSE, MAE)
- Mincer-Zarnowitz regression
- Vol-targeting leverage clipping
- Portfolio performance metrics

## References

- **Corsi, F. (2009)** — "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics* 7(2), 174-196.
- **Bollerslev, T. (1986)** — "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics* 31(3), 307-327.
- **Paleologo, G. (2021)** — *Advanced Portfolio Management: A Quant's Guide for Fundamental Investors*. Wiley. Chapter 6: Volatility Targeting.
- **RiskMetrics (1996)** — *Technical Document*. J.P. Morgan. EWMA with lambda=0.94.
- **Patton, A. (2011)** — "Volatility Forecast Comparison Using Imperfect Volatility Proxies." *Journal of Econometrics* 160(1), 246-256. (QLIKE loss function)
- **Mincer, J. & Zarnowitz, V. (1969)** — "The Evaluation of Economic Forecasts." *Economic Forecasts and Expectations*. (MZ regression)
