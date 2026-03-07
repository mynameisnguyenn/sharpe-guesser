# Empirical Asset Pricing via Machine Learning

**A simplified implementation inspired by Gu, Kelly, Xiu (2020)**

---

## Overview

This project implements a simplified version of the framework described in
"Empirical Asset Pricing via Machine Learning" (Gu, Kelly, Xiu, *Review of
Financial Studies*, 2020). The original paper is one of the most influential
recent papers in quantitative finance: it systematically compares machine
learning methods for predicting stock returns using a comprehensive set of
firm-level characteristics.

The paper's key finding, stated plainly: **tree-based models (random forests,
gradient-boosted trees) and neural networks substantially outperform linear
models at predicting individual stock returns out of sample.** Among individual
predictors, momentum signals and short-term reversal are the most important
features, while traditional value signals (like book-to-market) contribute less
than the literature long assumed. The economic gains are large -- long-short
portfolios sorted on ML predictions earn Sharpe ratios above 1.5, far exceeding
standard factor benchmarks.

Our implementation simplifies the original in several ways to make it
achievable as a portfolio project. We use S&P 500 stocks (via `yfinance`)
rather than the full CRSP universe, 7 base features + 21 pairwise interactions
instead of 94 + interactions, and three models (elastic net, random forest,
gradient boosted trees) rather than the full suite of neural networks. The core
ideas remain: we construct firm-level characteristics, train models with
expanding windows to prevent lookahead bias, form long-short decile portfolios,
and evaluate performance with out-of-sample R², Fama-French factor regressions,
statistical significance tests, and transaction cost analysis.

## Project Roadmap

- [x] Data pipeline: fetch returns and compute characteristics
- [x] Feature engineering: momentum, volatility, size, liquidity signals
- [x] Feature interactions: pairwise products (21 interaction terms)
- [x] Elastic net model with expanding window
- [x] Random forest model with expanding window
- [x] Gradient boosted trees with expanding window
- [x] Long-short decile portfolio construction
- [x] Compare to Fama-French 3-factor benchmark (alpha regression)
- [x] Feature importance analysis: which signals drive predictions?
- [x] Out-of-sample R² (the paper's statistical evaluation metric)
- [x] Statistical significance tests (t-test on L/S spread)
- [x] Turnover analysis and transaction cost estimation
- [ ] Write-up: findings and interpretation

## Project Structure

```
empirical_asset_pricing/
├── README.md
├── requirements.txt
├── run_pipeline.py               # End-to-end orchestration (run this)
├── data/
│   ├── __init__.py
│   └── fetch_data.py             # S&P 500 data + Fama-French factors
├── src/
│   ├── __init__.py
│   ├── features.py               # 7 base features + pairwise interactions
│   ├── models.py                 # Elastic net, random forest, GBT, expanding window
│   ├── portfolio.py              # Decile sorts, L/S construction, turnover, costs
│   └── evaluate.py               # OOS R², FF alpha, significance, plots
├── tests/
│   ├── __init__.py
│   └── test_features.py          # Unit tests for feature functions
└── results/                      # Output: plots, tables
```

## How to Run

```bash
# 1. Activate the virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (first run downloads ~500 stocks, cached after)
cd projects/empirical_asset_pricing
python run_pipeline.py
```

## Methodology

### Features (28 total)

**7 base features:** momentum (1m, 6m, 12m), realized volatility, log dollar
volume, volume trend, max return. These capture the signal groups that the
Kelly paper identifies as most important.

**21 interaction terms:** all pairwise products of base features. The paper's
key finding is that ML models capture feature interactions (e.g., "momentum
works differently for high-vol vs low-vol stocks") that linear models miss.
Adding explicit interactions gives the elastic net access to this signal.

### Models

| Model | Type | Key Property |
|---|---|---|
| Elastic Net | Penalized linear | L1 + L2 regularization, automatic feature selection |
| Random Forest | Tree ensemble (bagging) | Captures nonlinearity, robust to outliers |
| Gradient Boosting | Tree ensemble (boosting) | Sequential error correction, best performer in the paper |

### Evaluation

| Metric | What It Measures |
|---|---|
| OOS R² | Statistical predictive power (>0% beats naive forecast) |
| L/S Sharpe ratio | Economic value of predictions |
| Decile monotonicity | Whether rankings translate to ordered returns |
| FF 3-factor alpha | Alpha after controlling for market, size, value |
| t-statistic on spread | Statistical significance of L/S returns |
| Turnover & cost analysis | Whether alpha survives transaction costs |

### Walk-Forward Design

All predictions use an **expanding window** with `min_periods=36` months. At
each month t, the model trains on months 1 through t-1 and predicts month t.
The model never sees future data. This is the same walk-forward methodology
used in the Kelly paper and is standard practice for backtesting quantitative
strategies.

## Survivorship Bias Disclosure

**This analysis uses today's S&P 500 constituents.** This means we only include
companies that survived to the present day — stocks that were delisted, went
bankrupt, or were removed from the index (e.g., Lehman Brothers, Enron,
General Electric pre-2020) are excluded from the sample.

This introduces **survivorship bias**, which generally flatters results:
- The short side of our L/S portfolio doesn't include the worst-performing
  stocks (which were removed from the index before our analysis).
- The long side benefits from the fact that all stocks in our universe survived.

**How significant is this?** For the S&P 500 over 2010-2024, the effect is
moderate. The S&P 500 is a large-cap index with relatively low turnover (~20
names per year). A production system would track historical index membership
using CRSP data (which includes delisted stocks with proper return adjustments).

The Kelly paper uses the full CRSP universe including delisted stocks, which is
one reason their results are more robust. Our simplified approach is appropriate
for a portfolio project but would not be acceptable for a published paper or a
live trading strategy.

## Simplifications vs. the Original Paper

| Dimension | Kelly Paper | This Project |
|---|---|---|
| Universe | Full CRSP (~30,000 stocks) | S&P 500 (~500 stocks) |
| Base features | 94 characteristics | 7 characteristics |
| Interactions | ~4,400 terms | 21 terms (all pairwise) |
| Models | OLS, elastic net, PCA, PLS, RF, GBT, NN (1-5 layers) | Elastic net, RF, GBT |
| Data source | CRSP/Compustat (no survivorship bias) | yfinance (survivorship bias) |
| Period | 1957-2016 | 2010-2024 |
| Evaluation | OOS R², Sharpe, Diebold-Mariano test | OOS R², Sharpe, FF alpha, t-test |

## References

- **Gu, Kelly, Xiu (2020)** — "Empirical Asset Pricing via Machine Learning",
  *Review of Financial Studies* 33(5), 2223-2273.
  [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577)
- **IPCA** — Instrumented Principal Component Analysis (Kelly, Pruitt, Su).
  [GitHub: bkelly-lab/ipca](https://github.com/bkelly-lab/ipca)
- **bkelly-lab** — Bryan Kelly's research code.
  [GitHub: bkelly-lab](https://github.com/bkelly-lab)
- **Paleologo (2021)** — *Advanced Portfolio Management: A Quant's Guide for
  Fundamental Investors*. Wiley.
- **Fama, French (1993)** — "Common Risk Factors in the Returns on Stocks and
  Bonds", *Journal of Financial Economics* 33(1), 3-56.
- **Jegadeesh, Titman (1993)** — "Returns to Buying Winners and Selling Losers",
  *Journal of Finance* 48(1), 65-91.
