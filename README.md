# sharpe-guesser

Learn quantitative finance by building — starting with the Sharpe ratio.

## What's in here

| File / Directory | Purpose |
|------|---------|
| `sharpe_101.py` | Learn the Sharpe ratio from first principles with real market data |
| `sharpe_guesser.py` | Interactive game: guess the Sharpe ratio from a return chart |
| `factor_dashboard.py` | Factor exposure dashboard (CAPM, Fama-French, rolling beta) |
| `modules/` | 5 learning modules: statistics, risk metrics, factor models, optimization, strategies |
| `notebooks/factor_math_interactive.ipynb` | Interactive version of the factor math guide — tweak inputs, see charts |
| `notebooks/factor_models_explained.ipynb` | Factor models walkthrough with real market data |
| `projects/empirical_asset_pricing/` | ML return prediction (Gu, Kelly, Xiu 2020) — elastic net, random forest, long-short portfolios |
| `projects/risk_dashboard/` | Interactive Streamlit risk dashboard — VaR, factor exposure, rolling beta |
| `tests/` | Unit tests (96+) |
| `FACTOR_MATH_GUIDE.md` | Intuitive guide to factor model math |
| `LEARNING_ROADMAP.md` | Practical bootcamp roadmap |

## Quick start

```bash
pip install -r requirements.txt

# Learn the concepts
python sharpe_101.py

# Test your intuition
python sharpe_guesser.py --rounds 5
```

## Who this is for

Risk and data professionals who want to learn programming through
finance concepts they already understand.
