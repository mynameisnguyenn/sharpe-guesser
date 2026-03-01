# Quant Finance Learning Roadmap

A practical path from risk/data associate to quantitative finance — focused
on building things, not just reading theory.

## Where You Are Now

As a risk & data associate at a tiger cub, you already have:

- Exposure to portfolio construction, factor models, and risk reports
- Intuition for volatility regimes, drawdowns, and correlation
- Access to real PnL data and live trading context
- A network of PMs and quants you can learn from

That's a huge head start. Most quant aspirants have the math but not the
market intuition. You have the intuition — now you need the tools.

---

## Phase 1: Python + The Sharpe Ratio ← YOU ARE HERE

**Goal:** Get comfortable writing Python that does real financial computation.

| Topic | What to build | Key concepts |
|-------|--------------|--------------|
| Sharpe ratio | `sharpe_101.py` (this repo) | Returns, volatility, annualisation, risk-free rate |
| Sharpe guesser | `sharpe_guesser.py` (this repo) | Data fetching, visualisation, building intuition |

**Exercises:**
1. Run `sharpe_101.py` and modify it to analyse your fund's benchmark
2. Play `sharpe_guesser.py` until you can guess within 0.3 consistently
3. Add a new function to `sharpe_101.py` that computes the **Sortino ratio**
   (same as Sharpe but only penalises downside vol — your risk desk will
   appreciate that you know the difference)

**Python skills you'll pick up:** variables, functions, pandas, numpy, matplotlib, APIs.

---

## Phase 2: Risk Metrics & Factor Models

**Goal:** Automate the reports you already read manually.

| Topic | What to build | Key concepts |
|-------|--------------|--------------|
| VaR & CVaR | Historical and parametric VaR calculator | Distributions, quantiles, tail risk |
| Drawdown analysis | Max drawdown tracker with recovery periods | Time series, rolling windows |
| Beta & factor exposure | Regress returns against Fama-French factors | Linear regression, statsmodels, OLS |
| Correlation heatmaps | Dynamic correlation dashboard | Correlation matrices, rolling windows |

**Why this matters for you:** These are the reports your desk produces.
Building them from scratch means you understand what the numbers actually mean,
not just what cell to look at.

---

## Phase 3: Portfolio Optimisation

**Goal:** Go from "measuring risk" to "constructing portfolios."

| Topic | What to build | Key concepts |
|-------|--------------|--------------|
| Mean-variance optimisation | Efficient frontier plotter | Markowitz, convex optimisation |
| Risk parity | Equal risk contribution portfolio | Iterative solvers, scipy.optimize |
| Black-Litterman | Combine market equilibrium with PM views | Bayesian updating, prior/posterior |
| Backtesting framework | Simple event-driven backtester | OOP, event loops, transaction costs |

**Why this matters for you:** This is the bridge from risk to portfolio
management. If you can build a backtester, you can test ideas before
pitching them.

---

## Phase 4: Quantitative Strategies

**Goal:** Design, test, and evaluate systematic strategies.

| Topic | What to build | Key concepts |
|-------|--------------|--------------|
| Momentum | Cross-sectional momentum strategy | Signal construction, ranking, rebalancing |
| Mean reversion | Pairs trading with cointegration | Stationarity, Engle-Granger, z-scores |
| Factor investing | Multi-factor stock selection model | Fama-French, alpha, information ratio |
| Signal research | Feature importance & decay analysis | Pandas, scikit-learn, IC (information coefficient) |

---

## Phase 5: Advanced Topics (Pick Your Path)

At this point you'll know whether you want to be a:

### Path A: Systematic Quant (Strategy)
- Time series analysis (ARIMA, GARCH)
- Machine learning for alpha (tree models, not deep learning — start simple)
- Execution and market microstructure
- Alternative data (NLP on earnings calls, satellite data)

### Path B: Quant Risk (Your natural extension)
- Stress testing frameworks
- Greeks and options pricing (Black-Scholes, Monte Carlo)
- Counterparty credit risk models
- Regulatory capital (Basel III/IV)

### Path C: Quant Developer / Infrastructure
- Build production-grade data pipelines
- Real-time risk systems
- Low-latency Python (numba, cython) or switch to C++
- Database design for tick data (kdb+, Arctic, TimescaleDB)

---

## Recommended Resources

**Books (in order):**
1. *Python for Finance* — Yves Hilpisch (practical, code-heavy)
2. *Quantitative Risk Management* — McNeil, Frey, Embrechts (your risk background makes this accessible)
3. *Advances in Financial Machine Learning* — Marcos López de Prado (the modern quant bible)
4. *Active Portfolio Management* — Grinold & Kahn (the classic on alpha and IC)

**Practice:**
- This repo — keep adding to it
- Kaggle financial datasets
- QuantConnect / Zipline for backtesting
- Project Euler for pure programming skill

---

## The Real Advice

You don't need a PhD to be a quant. You need to:

1. **Build things** — every concept above has a "what to build" column for a reason
2. **Use your edge** — you sit next to quants and PMs every day; ask them what tools they wish existed
3. **Start with risk** — quant risk is the most natural transition from your current role, and it pays well
4. **Ship code at work** — automate one report, build one tool, volunteer for one data project. That's your real resume.

The Sharpe ratio is your first step. Everything else builds on it.
