# CLAUDE.md — Risk Dashboard

## What this is
An interactive Streamlit risk dashboard — a PORTFOLIO SHOWCASE piece that demonstrates
quantitative risk analysis skills. Self-contained (no imports from the learning modules).

## How to run
```bash
cd projects/risk_dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard structure (3 tabs)

### Tab 1: Risk Overview
- Per-ticker Sharpe, Sortino, Calmar ratios (st.metric cards)
- VaR/CVaR table (historical + parametric at 95% and 99%)
- Interactive drawdown chart with rebased price overlay (Plotly dual-axis)
- Expanders explaining each metric

### Tab 2: Factor Exposure
- CAPM regression table (alpha, beta, R-squared, t-stats)
- Fama-French 3-factor loadings table + grouped bar chart
- Information Ratio table
- Interactive rolling beta line chart
- Factor proxies: MKT=SPY, SMB=IWM-SPY, HML=IWD-IWF

### Tab 3: Comparison
- Side-by-side metrics table (all ratios, VaR, CVaR)
- Cumulative returns overlay (Plotly)
- Correlation heatmap (Plotly imshow)

## Sidebar controls
- Ticker input (comma-separated)
- Date range (start/end date pickers)
- Risk-free rate slider (0-10%, default 5%)
- Rolling window slider (21-252 days, default 63)
- VaR confidence level radio (95% or 99%)

## Architecture
- Single file: `app.py` (~470 lines)
- All calculations inline (portable, no external module imports)
- `@st.cache_data` for yfinance downloads
- Plotly for all interactive charts
- Graceful error handling for bad tickers / missing data

## Conventions
- Self-contained — replicates module_2 and module_3 logic but doesn't import from them
- Professional formatting: 2 decimal places for ratios, % for returns
- Educational expanders explaining what each metric means
- Wide layout (`st.set_page_config(layout="wide")`)
- TRADING_DAYS = 252

## Dependencies
streamlit, yfinance, plotly, pandas, numpy, scipy, statsmodels
