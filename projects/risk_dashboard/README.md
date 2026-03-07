# Interactive Risk Dashboard

**A Streamlit dashboard for portfolio risk analysis, factor decomposition, and comparative analytics.**

---

## Overview

A self-contained, interactive risk dashboard that computes the same metrics you'd see on a Bloomberg PORT screen or a hedge fund's internal risk report — but built from scratch in Python. Enter any tickers, adjust parameters, and immediately see risk metrics, factor exposures, and comparative analysis.

The dashboard is a single file (~470 lines) with all calculations inline — no external library dependencies beyond Streamlit, Plotly, and the standard quant stack. This makes it fully portable and easy to review.

## How to Run

```bash
cd projects/risk_dashboard
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens in your browser. Enter tickers (comma-separated), adjust the sidebar controls, and explore.

## Dashboard Layout

### Sidebar Controls

| Control | Default | Range | What It Does |
|---------|---------|-------|--------------|
| Tickers | AAPL, MSFT, JPM | Any valid ticker | Stocks to analyze |
| Start Date | 2020-01-01 | Any date | Analysis start |
| End Date | Today | Any date | Analysis end |
| Risk-Free Rate | 5% | 0-10% | For Sharpe/Sortino/Calmar calculations |
| Rolling Window | 63 days | 21-252 | For rolling beta and rolling vol |
| VaR Confidence | 95% | 95% or 99% | For VaR/CVaR calculations |

### Tab 1: Risk Overview

**Metric cards** (st.metric) for each ticker:
- **Sharpe Ratio** = (annualized excess return) / (annualized vol)
- **Sortino Ratio** = (annualized excess return) / (annualized downside deviation)
- **Calmar Ratio** = (annualized excess return) / |max drawdown|

**VaR/CVaR table:**
- **Historical VaR** = empirical quantile of returns at (1 - confidence)
- **Parametric VaR** = norm.ppf(1 - confidence, mu, sigma) — assumes normal returns
- **Historical CVaR** = mean of returns below the VaR threshold (expected shortfall)

**Drawdown chart** (Plotly, interactive):
- Rebased price line (cumulative return from $1)
- Drawdown fill below zero
- Max drawdown highlighted

**Educational expanders** explaining what each metric means in plain English.

### Tab 2: Factor Exposure

**CAPM regression table:**
```
R_stock - Rf = alpha + beta * (R_market - Rf) + epsilon
```
- Alpha (annualized), beta, R-squared, t-statistics for each ticker
- Alpha is the return your PM generated that ISN'T explained by the market

**Fama-French 3-factor loadings:**
```
R_stock - Rf = alpha + b1*(MKT-RF) + b2*SMB + b3*HML + epsilon
```
- Factor proxies: MKT = SPY, SMB = IWM - SPY, HML = IWD - IWF
- Grouped bar chart showing loadings per ticker
- Negative HML = growth stock, positive HML = value stock
- Positive SMB = small-cap tilt, negative SMB = large-cap tilt

**Information Ratio:**
- IR = alpha / tracking_error
- Benchmarks: > 0.5 is good, > 1.0 is exceptional

**Rolling beta chart** (Plotly, interactive):
- Shows how market sensitivity changes over time
- Useful for identifying regime changes (e.g., COVID spike in March 2020)

### Tab 3: Comparison

**Side-by-side metrics table:**
- All risk ratios, VaR, CVaR for every ticker in one table
- Easy to compare risk profiles across names

**Cumulative returns overlay:**
- All tickers on one chart, rebased to $1
- Quickly see relative performance

**Correlation heatmap:**
- Pairwise return correlations
- Helps identify diversification opportunities

## Frequently Asked Questions

### "What's the difference between VaR and CVaR?"

**VaR** answers: "What's the worst loss I expect on 95% of days?" It's a threshold — you lose MORE than VaR only 5% of the time (at 95% confidence).

**CVaR** (Conditional VaR, also called Expected Shortfall) answers: "When things are bad (worse than VaR), how bad on average?" It's the MEAN loss in the worst 5% of days. CVaR is always worse than VaR and is considered a better risk measure because it captures tail severity, not just tail probability.

### "Why is parametric VaR different from historical VaR?"

Parametric VaR assumes returns follow a normal distribution. Historical VaR uses the actual empirical distribution. In practice, stock returns have fatter tails than normal — so parametric VaR tends to UNDERSTATE the true risk. The gap between them tells you how non-normal your returns are.

### "What does a negative alpha mean?"

After controlling for factor exposure, this stock underperformed on a risk-adjusted basis. But check the t-statistic — if |t| < 2, the alpha isn't statistically significant, and it could just be noise.

### "Why does rolling beta change over time?"

Beta isn't a fixed number — it changes with market regimes. During crises (2008, March 2020), correlations spike and many stocks see higher betas. During calm markets, sector-specific factors matter more and betas diverge. Rolling beta shows you these dynamics.

### "What do the factor loadings tell me about a stock?"

- **High MKT beta (>1):** More volatile than the market. Tech stocks typically have beta > 1.
- **Positive SMB:** Behaves like a small-cap stock (even if it's technically large-cap).
- **Positive HML:** Behaves like a value stock. Negative HML = growth stock.
- **Significant alpha:** The stock generates returns beyond what the factors explain.

### "How is Sortino different from Sharpe?"

Sharpe penalizes ALL volatility equally — both upside and downside. Sortino only penalizes DOWNSIDE volatility (returns below zero). A stock with high upside volatility and low downside volatility will have a much better Sortino than Sharpe. Sortino is arguably a better metric because investors don't mind upside volatility.

### "What is Information Ratio?"

IR = alpha / tracking_error, where tracking_error is the vol of the residuals from the factor regression. It measures alpha per unit of idiosyncratic risk. Think of it as "how efficiently does this stock (or manager) convert active bets into alpha?"

The Fundamental Law of Active Management says: IR = IC * sqrt(breadth), where IC is the information coefficient (skill) and breadth is the number of independent bets. This is why systematic managers with many positions (high breadth) tend to have higher IRs than concentrated fundamental managers.

## Architecture

- **Single file:** `app.py` (~470 lines). All calculations inline — easy to audit and review.
- **Self-contained:** No imports from the learning modules. This is a portable portfolio piece.
- **Caching:** `@st.cache_data` for yfinance downloads — the dashboard stays responsive after initial load.
- **Error handling:** Bad tickers, missing data, and edge cases are handled gracefully with st.warning messages.

## Dependencies

```
streamlit
yfinance
plotly
pandas
numpy
scipy
statsmodels
```

## References

- Sharpe, W. (1966) — "Mutual Fund Performance." *Journal of Business*.
- Sortino, F. & van der Meer, R. (1991) — "Downside Risk." *Journal of Portfolio Management*.
- Fama, E. & French, K. (1993) — "Common Risk Factors in the Returns on Stocks and Bonds." *JFE*.
- Grinold, R. & Kahn, R. (2000) — *Active Portfolio Management*. (Information Ratio, Fundamental Law)
