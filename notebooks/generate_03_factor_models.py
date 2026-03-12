#!/usr/bin/env python3
"""Generate the 03_factor_models.ipynb notebook."""

import json

def md(source_lines):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }

def code(source_lines):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source_lines,
        "execution_count": None,
        "outputs": [],
    }

cells = []

# =============================================================================
# Title
# =============================================================================
cells.append(md([
    "# Module 3: Factor Models & Regression\n",
    "\n",
    "This notebook teaches you the **factor model** framework that underpins\n",
    "modern portfolio management: CAPM, Fama-French, rolling beta, alpha\n",
    "estimation, and the Information Ratio.\n",
    "\n",
    "**Why this matters:** Every portfolio manager's return can be decomposed into\n",
    "*factor exposure* (beta you can get cheaply with ETFs) and *alpha*\n",
    "(genuine skill). Understanding this decomposition is how risk teams evaluate PMs,\n",
    "how allocators pick funds, and how you'll build signals later in this course.\n",
    "\n",
    "**How to use:** Run each cell with Shift+Enter. Read the explanations,\n",
    "then modify tickers to build intuition.\n",
    "\n",
    "---"
]))

# =============================================================================
# Setup cell
# =============================================================================
cells.append(code([
    "# Setup — run this first\n",
    "import sys, os\n",
    "sys.path.insert(0, os.path.abspath(\"..\"))\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import statsmodels.api as sm\n",
    "from scipy import stats\n",
    "from data_loader import fetch_prices, fetch_multi_prices\n",
    "\n",
    "%matplotlib inline\n",
    "plt.rcParams[\"figure.figsize\"] = (12, 5)\n",
    "TRADING_DAYS = 252\n",
    "print(\"Ready.\")"
]))

# =============================================================================
# Fetch data
# =============================================================================
cells.append(md([
    "## Fetch Data\n",
    "\n",
    "We need several tickers for this module:\n",
    "- **AAPL** — the stock we want to analyse\n",
    "- **SPY** — market proxy ($R_m$)\n",
    "- **IWM** — small-cap proxy (Russell 2000 ETF)\n",
    "- **IWD** — value proxy (Russell 1000 Value ETF)\n",
    "- **IWF** — growth proxy (Russell 1000 Growth ETF)\n",
    "- **JPM, XOM** — for cross-asset comparison later"
]))

cells.append(code([
    "# ============================\n",
    "# CHANGE THESE\n",
    "# ============================\n",
    "STOCK  = \"AAPL\"\n",
    "START  = \"2018-01-01\"\n",
    "END    = \"2025-12-31\"\n",
    "\n",
    "tickers = [STOCK, \"SPY\", \"IWM\", \"IWD\", \"IWF\", \"JPM\", \"XOM\"]\n",
    "prices = fetch_multi_prices(tickers, START, END)\n",
    "returns = prices.pct_change().dropna()\n",
    "\n",
    "print(f\"Loaded {len(prices)} days for {len(tickers)} tickers\")\n",
    "print(f\"Date range: {prices.index[0].date()} to {prices.index[-1].date()}\")\n",
    "returns.head()"
]))

# =============================================================================
# Section 1: CAPM
# =============================================================================
cells.append(md([
    "---\n",
    "## 1. CAPM (Capital Asset Pricing Model)\n",
    "\n",
    "The Capital Asset Pricing Model says that the **excess return** of any asset\n",
    "is explained by its exposure to the **market's excess return**:\n",
    "\n",
    "$$R_i - R_f = \\alpha_i + \\beta_i (R_m - R_f) + \\epsilon_i$$\n",
    "\n",
    "where:\n",
    "- $R_i$ = return of asset $i$\n",
    "- $R_f$ = risk-free rate (we approximate as 0 for simplicity)\n",
    "- $R_m$ = return of the market (SPY)\n",
    "- $\\beta_i$ = sensitivity of the asset to the market\n",
    "- $\\alpha_i$ = excess return not explained by market exposure\n",
    "- $\\epsilon_i$ = residual (idiosyncratic noise)\n",
    "\n",
    "### Interpreting Beta\n",
    "\n",
    "| Beta | Meaning | Example |\n",
    "|------|---------|--------|\n",
    "| $\\beta > 1$ | More volatile than the market | Tech stocks (AAPL, NVDA) |\n",
    "| $\\beta = 1$ | Moves with the market | SPY itself |\n",
    "| $\\beta < 1$ | Less volatile than the market | Utilities (XLU), Consumer Staples (XLP) |\n",
    "| $\\beta < 0$ | Moves opposite to market | Rare — some gold miners, VIX products |\n",
    "\n",
    "### Interpreting Alpha\n",
    "\n",
    "- $\\alpha > 0$: the asset earns **more** than its beta would predict (skill or anomaly)\n",
    "- $\\alpha = 0$: returns are fully explained by market exposure\n",
    "- $\\alpha < 0$: the asset **underperforms** after adjusting for market risk\n",
    "\n",
    "Alpha is what every PM is trying to generate. Beta is what you get for free by buying an index fund."
]))

cells.append(code([
    "# CAPM regression: AAPL excess return vs SPY excess return\n",
    "# We approximate Rf = 0 for simplicity (typical for short horizons)\n",
    "\n",
    "y = returns[STOCK]            # dependent variable: stock return\n",
    "X = returns[\"SPY\"]            # independent variable: market return\n",
    "X_const = sm.add_constant(X)  # add intercept (alpha)\n",
    "\n",
    "capm_model = sm.OLS(y, X_const).fit()\n",
    "\n",
    "alpha_daily = capm_model.params.iloc[0]\n",
    "beta = capm_model.params.iloc[1]\n",
    "r_squared = capm_model.rsquared\n",
    "\n",
    "print(f\"CAPM Regression: {STOCK} vs SPY\")\n",
    "print(f\"{'='*50}\")\n",
    "print(f\"  Alpha (daily):      {alpha_daily:.6f}\")\n",
    "print(f\"  Alpha (annualised): {alpha_daily * 252:.2%}\")\n",
    "print(f\"  Beta:               {beta:.4f}\")\n",
    "print(f\"  R-squared:          {r_squared:.4f}\")\n",
    "print()\n",
    "print(f\"  t-stat (alpha):     {capm_model.tvalues.iloc[0]:.2f}  \"\n",
    "      f\"(p={capm_model.pvalues.iloc[0]:.4f})\")\n",
    "print(f\"  t-stat (beta):      {capm_model.tvalues.iloc[1]:.2f}  \"\n",
    "      f\"(p={capm_model.pvalues.iloc[1]:.4f})\")\n",
    "print()\n",
    "print(f\"Interpretation:\")\n",
    "if beta > 1:\n",
    "    print(f\"  Beta = {beta:.2f} > 1: {STOCK} is MORE volatile than the market.\")\n",
    "elif beta < 1:\n",
    "    print(f\"  Beta = {beta:.2f} < 1: {STOCK} is LESS volatile than the market.\")\n",
    "else:\n",
    "    print(f\"  Beta = {beta:.2f} = 1: {STOCK} moves with the market.\")\n",
    "print(f\"  R-squared = {r_squared:.1%}: market explains {r_squared:.0%} \"\n",
    "      f\"of {STOCK}'s return variance.\")\n",
    "print(f\"  The remaining {1-r_squared:.0%} is idiosyncratic (stock-specific).\")"
]))

cells.append(md([
    "### CAPM Scatter Plot\n",
    "\n",
    "Each dot is one day. The slope of the best-fit line is beta."
]))

cells.append(code([
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "\n",
    "ax.scatter(returns[\"SPY\"], returns[STOCK], alpha=0.3, s=10, color=\"steelblue\")\n",
    "\n",
    "# Best-fit line\n",
    "x_line = np.linspace(returns[\"SPY\"].min(), returns[\"SPY\"].max(), 100)\n",
    "y_line = alpha_daily + beta * x_line\n",
    "ax.plot(x_line, y_line, \"r-\", linewidth=2,\n",
    "        label=f\"$R_{{AAPL}}$ = {alpha_daily:.5f} + {beta:.2f} $\\\\times R_{{SPY}}$\")\n",
    "\n",
    "ax.axhline(0, color=\"grey\", linewidth=0.5)\n",
    "ax.axvline(0, color=\"grey\", linewidth=0.5)\n",
    "ax.set_xlabel(\"SPY Daily Return\")\n",
    "ax.set_ylabel(f\"{STOCK} Daily Return\")\n",
    "ax.set_title(f\"CAPM: {STOCK} vs SPY\")\n",
    "ax.legend(fontsize=11)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"The slope (beta = {beta:.2f}) tells you: when SPY moves 1%, \")\n",
    "print(f\"{STOCK} moves ~{beta:.2f}% on average.\")"
]))

# =============================================================================
# Section 2: Fama-French 3-Factor
# =============================================================================
cells.append(md([
    "---\n",
    "## 2. Fama-French 3-Factor Model\n",
    "\n",
    "CAPM says all risk comes from one source: the market. Fama and French (1993)\n",
    "showed that **two additional factors** help explain the cross-section of returns:\n",
    "\n",
    "$$R_i - R_f = \\alpha_i + \\beta_1(R_m - R_f) + \\beta_2 \\cdot SMB + \\beta_3 \\cdot HML + \\epsilon_i$$\n",
    "\n",
    "### The Three Factors\n",
    "\n",
    "| Factor | Name | Long | Short | Rationale |\n",
    "|--------|------|------|-------|-----------|\n",
    "| $R_m - R_f$ | **Market** | Equities | Risk-free | Equity risk premium |\n",
    "| $SMB$ | **Small Minus Big** | Small caps | Large caps | Small firms earn a premium for illiquidity/risk |\n",
    "| $HML$ | **High Minus Low** | Value (high B/M) | Growth (low B/M) | Cheap stocks outperform expensive ones on average |\n",
    "\n",
    "### ETF Proxies\n",
    "\n",
    "In practice, we approximate these factors using liquid ETFs:\n",
    "- **MKT** = SPY (S&P 500)\n",
    "- **SMB** = IWM $-$ SPY (small cap minus large cap)\n",
    "- **HML** = IWD $-$ IWF (value minus growth)\n",
    "\n",
    "### What the Loadings Tell You\n",
    "\n",
    "- $\\beta_2 > 0$: the stock behaves like a **small cap** (more exposure to size risk)\n",
    "- $\\beta_2 < 0$: the stock behaves like a **large cap**\n",
    "- $\\beta_3 > 0$: the stock behaves like a **value stock**\n",
    "- $\\beta_3 < 0$: the stock behaves like a **growth stock**"
]))

cells.append(code([
    "# Construct Fama-French factors from ETF proxies\n",
    "factors = pd.DataFrame({\n",
    "    \"MKT\": returns[\"SPY\"],\n",
    "    \"SMB\": returns[\"IWM\"] - returns[\"SPY\"],     # small minus big\n",
    "    \"HML\": returns[\"IWD\"] - returns[\"IWF\"],      # value minus growth\n",
    "}).dropna()\n",
    "\n",
    "# Align stock returns with factor dates\n",
    "y_ff = returns[STOCK].loc[factors.index]\n",
    "\n",
    "# Run multi-factor regression\n",
    "X_ff = sm.add_constant(factors)\n",
    "ff_model = sm.OLS(y_ff, X_ff).fit()\n",
    "\n",
    "print(f\"Fama-French 3-Factor Regression: {STOCK}\")\n",
    "print(f\"{'='*60}\")\n",
    "print()\n",
    "\n",
    "# Display results in a clean table\n",
    "results = pd.DataFrame({\n",
    "    \"Coefficient\": ff_model.params,\n",
    "    \"t-statistic\": ff_model.tvalues,\n",
    "    \"p-value\": ff_model.pvalues,\n",
    "    \"Significant?\": [\"***\" if p < 0.01 else \"**\" if p < 0.05 \n",
    "                     else \"*\" if p < 0.1 else \"\" \n",
    "                     for p in ff_model.pvalues]\n",
    "})\n",
    "print(results.round(4).to_string())\n",
    "print()\n",
    "print(f\"  R-squared:          {ff_model.rsquared:.4f}\")\n",
    "print(f\"  Adj. R-squared:     {ff_model.rsquared_adj:.4f}\")\n",
    "print(f\"  Alpha (annualised): {ff_model.params.iloc[0] * 252:.2%}\")\n",
    "print()\n",
    "print(f\"Interpretation:\")\n",
    "smb_coef = ff_model.params[\"SMB\"]\n",
    "hml_coef = ff_model.params[\"HML\"]\n",
    "print(f\"  MKT loading = {ff_model.params['MKT']:.2f}: \"\n",
    "      f\"market beta after controlling for size & value\")\n",
    "print(f\"  SMB loading = {smb_coef:.2f}: \"\n",
    "      f\"{STOCK} behaves like a {'small' if smb_coef > 0 else 'large'} cap\")\n",
    "print(f\"  HML loading = {hml_coef:.2f}: \"\n",
    "      f\"{STOCK} behaves like a {'value' if hml_coef > 0 else 'growth'} stock\")"
]))

cells.append(md([
    "### Factor Loadings — Visual Comparison\n",
    "\n",
    "A bar chart makes it easy to see which factors dominate."
]))

cells.append(code([
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "\n",
    "factor_names = [\"MKT\", \"SMB\", \"HML\"]\n",
    "loadings = [ff_model.params[f] for f in factor_names]\n",
    "colors = [\"steelblue\" if v >= 0 else \"salmon\" for v in loadings]\n",
    "\n",
    "bars = ax.bar(factor_names, loadings, color=colors, edgecolor=\"black\", linewidth=0.5)\n",
    "ax.axhline(0, color=\"black\", linewidth=0.5)\n",
    "ax.set_ylabel(\"Factor Loading (Beta)\")\n",
    "ax.set_title(f\"{STOCK} — Fama-French 3-Factor Loadings\")\n",
    "\n",
    "for bar, val in zip(bars, loadings):\n",
    "    ax.text(bar.get_x() + bar.get_width() / 2, val,\n",
    "            f\"{val:.3f}\", ha=\"center\",\n",
    "            va=\"bottom\" if val >= 0 else \"top\", fontsize=12)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# =============================================================================
# Section 3: Rolling Beta
# =============================================================================
cells.append(md([
    "---\n",
    "## 3. Rolling Beta\n",
    "\n",
    "The CAPM regression above estimates a **single beta** over the entire sample.\n",
    "But beta is **not constant** — it drifts over time with:\n",
    "- Market regimes (bull vs bear)\n",
    "- Earnings surprises\n",
    "- Sector rotation\n",
    "- Changes in a company's leverage or business mix\n",
    "\n",
    "### Rolling Beta Formula\n",
    "\n",
    "We compute beta over a rolling window of $w$ days:\n",
    "\n",
    "$$\\beta_t = \\frac{\\text{Cov}(R_i, R_m)_t}{\\text{Var}(R_m)_t}$$\n",
    "\n",
    "where the covariance and variance are computed over the window $[t-w+1, t]$.\n",
    "\n",
    "### Common Window Choices\n",
    "\n",
    "| Window | Days | Use Case |\n",
    "|--------|------|----------|\n",
    "| 1 month | 21 | Very noisy, but responsive |\n",
    "| 3 months | 63 | Good balance of responsiveness and stability |\n",
    "| 6 months | 126 | Smoother, used for risk reports |\n",
    "| 1 year | 252 | Very smooth, used for strategic allocation |\n",
    "\n",
    "63 days (one quarter) is the most common choice for equity risk models."
]))

cells.append(code([
    "# Rolling beta: 63-day (one quarter) window\n",
    "WINDOW = 63\n",
    "\n",
    "rolling_cov = returns[STOCK].rolling(WINDOW).cov(returns[\"SPY\"])\n",
    "rolling_var = returns[\"SPY\"].rolling(WINDOW).var()\n",
    "rolling_beta = rolling_cov / rolling_var\n",
    "\n",
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)\n",
    "\n",
    "# Top: price\n",
    "ax1.plot(prices[STOCK], linewidth=0.8, color=\"steelblue\")\n",
    "ax1.set_ylabel(\"Price ($)\")\n",
    "ax1.set_title(f\"{STOCK} — Price\")\n",
    "\n",
    "# Bottom: rolling beta\n",
    "ax2.plot(rolling_beta, linewidth=1, color=\"darkred\")\n",
    "ax2.axhline(beta, color=\"grey\", ls=\"--\", linewidth=0.8,\n",
    "            label=f\"Full-sample beta = {beta:.2f}\")\n",
    "ax2.axhline(1.0, color=\"black\", ls=\":\", linewidth=0.5, label=\"Beta = 1\")\n",
    "ax2.set_ylabel(\"Rolling Beta\")\n",
    "ax2.set_title(f\"{STOCK} — {WINDOW}-Day Rolling Beta vs SPY\")\n",
    "ax2.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"Rolling beta range: {rolling_beta.min():.2f} to {rolling_beta.max():.2f}\")\n",
    "print(f\"Full-sample beta:   {beta:.2f}\")\n",
    "print()\n",
    "print(\"Key insight: beta is NOT a constant. It drifts with regimes,\")\n",
    "print(\"earnings, and sector rotation. Risk models that use a static\")\n",
    "print(\"beta can significantly mis-estimate portfolio risk.\")"
]))

# =============================================================================
# Section 4: Alpha & Residual Analysis
# =============================================================================
cells.append(md([
    "---\n",
    "## 4. Alpha & Residual Analysis\n",
    "\n",
    "The **residuals** from the CAPM (or Fama-French) regression are the\n",
    "**idiosyncratic returns** — what is left after stripping out factor exposure:\n",
    "\n",
    "$$\\epsilon_t = R_{i,t} - (\\hat{\\alpha} + \\hat{\\beta} R_{m,t})$$\n",
    "\n",
    "### Why Residuals Matter\n",
    "\n",
    "- Residuals represent the **stock-specific** component of returns\n",
    "- If residuals are random noise (no pattern), then alpha = 0 and the factor\n",
    "  model explains everything\n",
    "- If residuals have **structure** (trends, autocorrelation, predictable patterns),\n",
    "  there may be **alpha to capture**\n",
    "\n",
    "### What to Check\n",
    "\n",
    "1. **Time series plot**: do residuals cluster or trend?\n",
    "2. **Histogram**: are they normally distributed?\n",
    "3. **Autocorrelation**: does today's residual predict tomorrow's? (If yes, that's exploitable alpha.)"
]))

cells.append(code([
    "# Get CAPM residuals\n",
    "residuals = capm_model.resid\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 9))\n",
    "\n",
    "# 1. Residuals over time\n",
    "axes[0, 0].plot(residuals.index, residuals.values, linewidth=0.5,\n",
    "                color=\"steelblue\", alpha=0.7)\n",
    "axes[0, 0].axhline(0, color=\"red\", linewidth=0.8)\n",
    "axes[0, 0].set_title(f\"{STOCK} CAPM Residuals Over Time\")\n",
    "axes[0, 0].set_ylabel(\"Residual Return\")\n",
    "\n",
    "# 2. Histogram of residuals\n",
    "axes[0, 1].hist(residuals, bins=80, density=True, alpha=0.6,\n",
    "                color=\"steelblue\", edgecolor=\"white\")\n",
    "# Overlay normal fit\n",
    "mu_r, std_r = residuals.mean(), residuals.std()\n",
    "x_norm = np.linspace(mu_r - 4*std_r, mu_r + 4*std_r, 200)\n",
    "axes[0, 1].plot(x_norm, stats.norm.pdf(x_norm, mu_r, std_r),\n",
    "                \"r-\", linewidth=2, label=\"Normal fit\")\n",
    "axes[0, 1].set_title(\"Residual Distribution\")\n",
    "axes[0, 1].set_xlabel(\"Residual\")\n",
    "axes[0, 1].legend()\n",
    "\n",
    "# 3. Autocorrelation\n",
    "from statsmodels.graphics.tsaplots import plot_acf\n",
    "plot_acf(residuals.dropna(), lags=20, ax=axes[1, 0], alpha=0.05)\n",
    "axes[1, 0].set_title(\"Residual Autocorrelation (ACF)\")\n",
    "\n",
    "# 4. QQ plot\n",
    "stats.probplot(residuals.dropna(), dist=\"norm\", plot=axes[1, 1])\n",
    "axes[1, 1].set_title(\"Residual QQ Plot\")\n",
    "axes[1, 1].get_lines()[0].set_markerfacecolor(\"steelblue\")\n",
    "axes[1, 1].get_lines()[0].set_markersize(3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Durbin-Watson test for autocorrelation\n",
    "from statsmodels.stats.stattools import durbin_watson\n",
    "dw = durbin_watson(residuals)\n",
    "print(f\"Durbin-Watson statistic: {dw:.3f}\")\n",
    "print(f\"  (DW ~ 2.0 = no autocorrelation; DW < 2 = positive; DW > 2 = negative)\")\n",
    "print()\n",
    "print(f\"Residual skewness:       {residuals.skew():.3f}\")\n",
    "print(f\"Residual excess kurtosis: {residuals.kurtosis():.3f}\")\n",
    "print()\n",
    "if abs(dw - 2) < 0.3:\n",
    "    print(\"Residuals show little autocorrelation — hard to predict.\")\n",
    "else:\n",
    "    print(\"Residuals show some autocorrelation — there may be exploitable alpha.\")"
]))

# =============================================================================
# Section 5: Information Ratio
# =============================================================================
cells.append(md([
    "---\n",
    "## 5. Information Ratio\n",
    "\n",
    "The **Information Ratio** (IR) measures **alpha per unit of active risk**.\n",
    "It is the single most important metric for evaluating a portfolio manager's\n",
    "skill at generating returns beyond factor exposure.\n",
    "\n",
    "### Formula\n",
    "\n",
    "$$IR = \\frac{\\alpha_{\\text{annual}}}{\\sigma_{\\text{residuals}} \\cdot \\sqrt{252}}$$\n",
    "\n",
    "where:\n",
    "- $\\alpha_{\\text{annual}}$ = annualised alpha from the factor regression\n",
    "- $\\sigma_{\\text{residuals}} \\cdot \\sqrt{252}$ = annualised tracking error (volatility of residuals)\n",
    "\n",
    "### Benchmarks\n",
    "\n",
    "| IR | Quality |\n",
    "|----|--------|\n",
    "| $< 0$ | Negative alpha — destroying value |\n",
    "| $0.0 - 0.2$ | Weak — barely above noise |\n",
    "| $0.2 - 0.5$ | Decent — some evidence of skill |\n",
    "| $0.5 - 1.0$ | Good — consistently adding value |\n",
    "| $> 1.0$ | Exceptional — very rare, often unsustainable |\n",
    "\n",
    "### The Fundamental Law of Active Management\n",
    "\n",
    "Grinold & Kahn showed that:\n",
    "\n",
    "$$IR = IC \\times \\sqrt{\\text{Breadth}}$$\n",
    "\n",
    "where:\n",
    "- **IC** (Information Coefficient) = correlation between your forecasts and actual outcomes\n",
    "- **Breadth** = number of independent bets per year\n",
    "\n",
    "This is powerful: even a small IC (say 0.05) can produce a good IR if you make\n",
    "enough independent bets. A stat-arb fund with IC = 0.02 and 10,000 trades/year\n",
    "has IR = $0.02 \\times \\sqrt{10000} = 2.0$ — exceptional.\n",
    "\n",
    "A concentrated long-only PM with IC = 0.10 and 20 positions has\n",
    "IR = $0.10 \\times \\sqrt{20} = 0.45$ — decent but not great."
]))

cells.append(code([
    "# Compute Information Ratio from CAPM residuals\n",
    "alpha_annual = alpha_daily * 252\n",
    "tracking_error = residuals.std() * np.sqrt(252)\n",
    "information_ratio = alpha_annual / tracking_error\n",
    "\n",
    "print(f\"Information Ratio: {STOCK}\")\n",
    "print(f\"{'='*50}\")\n",
    "print(f\"  Alpha (daily):          {alpha_daily:.6f}\")\n",
    "print(f\"  Alpha (annualised):     {alpha_annual:.2%}\")\n",
    "print(f\"  Residual vol (daily):   {residuals.std():.4%}\")\n",
    "print(f\"  Tracking error (annual): {tracking_error:.2%}\")\n",
    "print(f\"  Information Ratio:      {information_ratio:.3f}\")\n",
    "print()\n",
    "\n",
    "if information_ratio > 1.0:\n",
    "    quality = \"EXCEPTIONAL (> 1.0)\"\n",
    "elif information_ratio > 0.5:\n",
    "    quality = \"Good (0.5 - 1.0)\"\n",
    "elif information_ratio > 0.2:\n",
    "    quality = \"Decent (0.2 - 0.5)\"\n",
    "elif information_ratio > 0:\n",
    "    quality = \"Weak (0.0 - 0.2)\"\n",
    "else:\n",
    "    quality = \"Negative alpha\"\n",
    "print(f\"  Quality: {quality}\")\n",
    "print()\n",
    "print(\"Note: this is the IR of HOLDING the stock vs the market,\")\n",
    "print(\"not the IR of a trading strategy. A fund PM's IR would\")\n",
    "print(\"be computed on the portfolio's active returns vs benchmark.\")"
]))

# =============================================================================
# Section 6: Cross-Asset Comparison
# =============================================================================
cells.append(md([
    "---\n",
    "## 6. Cross-Asset Comparison: AAPL vs JPM vs XOM\n",
    "\n",
    "Let's compare three stocks from different sectors to see how factor\n",
    "exposures differ:\n",
    "- **AAPL** — Technology (growth, large cap)\n",
    "- **JPM** — Financials (value tilt, rate-sensitive)\n",
    "- **XOM** — Energy (commodity exposure, value tilt)"
]))

cells.append(code([
    "# Run Fama-French regression for each stock\n",
    "comparison_stocks = [\"AAPL\", \"JPM\", \"XOM\"]\n",
    "comparison_results = {}\n",
    "\n",
    "for stock in comparison_stocks:\n",
    "    y_s = returns[stock].loc[factors.index]\n",
    "    model_s = sm.OLS(y_s, X_ff).fit()\n",
    "    comparison_results[stock] = {\n",
    "        \"Alpha (ann)\": model_s.params.iloc[0] * 252,\n",
    "        \"MKT Beta\": model_s.params[\"MKT\"],\n",
    "        \"SMB\": model_s.params[\"SMB\"],\n",
    "        \"HML\": model_s.params[\"HML\"],\n",
    "        \"R-squared\": model_s.rsquared,\n",
    "        \"IR\": (model_s.params.iloc[0] * 252) / \n",
    "              (model_s.resid.std() * np.sqrt(252)),\n",
    "    }\n",
    "\n",
    "comp_df = pd.DataFrame(comparison_results).T\n",
    "print(\"Factor Exposure Comparison\")\n",
    "print(\"=\" * 65)\n",
    "print(comp_df.round(4).to_string())"
]))

cells.append(code([
    "# Visual comparison of factor loadings\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "\n",
    "factor_names = [\"MKT Beta\", \"SMB\", \"HML\"]\n",
    "titles = [\"Market Beta\", \"Size (SMB)\", \"Value (HML)\"]\n",
    "\n",
    "for idx, (factor, title) in enumerate(zip(factor_names, titles)):\n",
    "    values = [comparison_results[s][factor] for s in comparison_stocks]\n",
    "    colors = [\"steelblue\" if v >= 0 else \"salmon\" for v in values]\n",
    "    bars = axes[idx].bar(comparison_stocks, values, color=colors,\n",
    "                         edgecolor=\"black\", linewidth=0.5)\n",
    "    axes[idx].axhline(0, color=\"black\", linewidth=0.5)\n",
    "    axes[idx].set_title(title)\n",
    "    axes[idx].set_ylabel(\"Loading\")\n",
    "    for bar, val in zip(bars, values):\n",
    "        axes[idx].text(bar.get_x() + bar.get_width() / 2, val,\n",
    "                       f\"{val:.3f}\", ha=\"center\",\n",
    "                       va=\"bottom\" if val >= 0 else \"top\", fontsize=10)\n",
    "\n",
    "plt.suptitle(\"Fama-French Factor Loadings by Stock\", fontsize=14, y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"Observations:\")\n",
    "print(\"  - AAPL: high market beta, negative HML = classic growth stock\")\n",
    "print(\"  - JPM:  financial sector, sensitive to market, may have value tilt\")\n",
    "print(\"  - XOM:  energy sector, lower market beta, positive HML = value stock\")"
]))

cells.append(code([
    "# Rolling beta comparison\n",
    "fig, ax = plt.subplots(figsize=(12, 5))\n",
    "\n",
    "for stock, color in zip(comparison_stocks, [\"steelblue\", \"darkred\", \"forestgreen\"]):\n",
    "    rc = returns[stock].rolling(WINDOW).cov(returns[\"SPY\"])\n",
    "    rv = returns[\"SPY\"].rolling(WINDOW).var()\n",
    "    rb = rc / rv\n",
    "    ax.plot(rb, linewidth=1, color=color, label=stock, alpha=0.8)\n",
    "\n",
    "ax.axhline(1.0, color=\"black\", ls=\":\", linewidth=0.5)\n",
    "ax.set_ylabel(\"Rolling Beta\")\n",
    "ax.set_title(f\"{WINDOW}-Day Rolling Beta vs SPY\")\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"Notice how betas diverge during market stress (2020, 2022).\")\n",
    "print(\"A risk model with static betas would mis-estimate hedging ratios.\")"
]))

# =============================================================================
# Exercises
# =============================================================================
cells.append(md([
    "---\n",
    "## Exercises\n",
    "\n",
    "1. **Change STOCK to MSFT** at the top and re-run. How does its beta compare\n",
    "   to AAPL? Is it more or less of a growth stock (HML loading)?\n",
    "\n",
    "2. **Try a utility stock (XLU).** What beta do you get? Is HML positive or\n",
    "   negative? Does this match your intuition about utilities being defensive/value?\n",
    "\n",
    "3. **Change the rolling window to 21 days (1 month).** How does the rolling beta\n",
    "   plot change? Is it more noisy? When would you prefer a short vs long window?\n",
    "\n",
    "4. **Fundamental Law exercise:** Suppose you have IC = 0.03 and make 500\n",
    "   independent bets per year. What is your expected IR? If you increased to\n",
    "   2000 bets, how does IR change? What does this say about high-frequency\n",
    "   vs low-frequency strategies?\n",
    "\n",
    "5. **Alpha significance:** Look at the CAPM regression t-stat for alpha.\n",
    "   Is it statistically significant (|t| > 2)? If not, what does that tell\n",
    "   you about whether the stock truly has alpha vs just noise?"
]))

cells.append(code([
    "# Exercise 4 starter code\n",
    "IC = 0.03\n",
    "breadth_low = 500\n",
    "breadth_high = 2000\n",
    "\n",
    "IR_low = IC * np.sqrt(breadth_low)\n",
    "IR_high = IC * np.sqrt(breadth_high)\n",
    "\n",
    "print(f\"Fundamental Law of Active Management\")\n",
    "print(f\"{'='*45}\")\n",
    "print(f\"  IC = {IC}\")\n",
    "print(f\"  Breadth = {breadth_low:>5d}  =>  IR = {IR_low:.2f}\")\n",
    "print(f\"  Breadth = {breadth_high:>5d}  =>  IR = {IR_high:.2f}\")\n",
    "print()\n",
    "print(f\"Doubling breadth from {breadth_low} to {breadth_high} \")\n",
    "print(f\"increases IR by {IR_high/IR_low:.1f}x (because sqrt(4) = 2).\")\n",
    "print()\n",
    "print(\"This is why quant funds focus on BREADTH — many small bets\")\n",
    "print(\"with mediocre IC can beat concentrated funds with high IC.\")"
]))

# =============================================================================
# Build the notebook
# =============================================================================
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
            "mimetype": "text/x-python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": cells,
}

output_path = "/home/user/sharpe-guesser/notebooks/03_factor_models.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Wrote {output_path}")
print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type']=='code')} code)")
