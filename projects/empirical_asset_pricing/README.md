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
rather than the full CRSP universe, roughly 15 features instead of 94+
interactions, and two models (elastic net and random forest) rather than the
full suite of neural networks and ensemble methods. The core ideas remain: we
construct firm-level characteristics, train models with expanding windows to
prevent lookahead bias, form long-short decile portfolios, and evaluate
performance against standard factor benchmarks.

## Project Roadmap

- [ ] Data pipeline: fetch returns and compute characteristics
- [ ] Feature engineering: momentum, volatility, size, value, liquidity signals
- [ ] Elastic net model with expanding window
- [ ] Random forest model with expanding window
- [ ] Long-short decile portfolio construction
- [ ] Compare to Fama-French 3-factor benchmark
- [ ] Feature importance analysis: which signals drive predictions?
- [ ] Write-up: findings and interpretation

## Project Structure

```
empirical_asset_pricing/
├── README.md
├── requirements.txt
├── data/
│   ├── __init__.py
│   └── fetch_data.py          # Data pipeline: S&P 500 via yfinance
├── src/
│   ├── __init__.py
│   ├── features.py            # Feature construction (momentum, vol, size, etc.)
│   ├── models.py              # Elastic net, random forest, expanding window
│   ├── portfolio.py           # Decile sorts, long-short construction
│   └── evaluate.py            # Performance metrics, plots, comparisons
├── notebooks/                 # Jupyter notebooks for exploration
├── tests/
│   ├── __init__.py
│   └── test_features.py       # Unit tests for feature functions
└── results/                   # Output: plots, tables, saved models
```

## How to Run

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch and cache data (takes a few minutes the first time)
python -m data.fetch_data

# 4. Run from the project root (projects/empirical_asset_pricing/)
#    Individual modules can be imported:
#      from src.features import build_features
#      from src.models import expanding_window_predict
```

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
