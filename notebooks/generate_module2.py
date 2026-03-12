"""
Generate 02_risk_metrics.ipynb — Module 2: Risk Metrics
========================================================
Run:  python generate_module2.py
"""
import json

cells = []


def md(source_lines):
    """Add a markdown cell."""
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    })


def code(source_lines):
    """Add a code cell."""
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": source_lines,
        "execution_count": None,
        "outputs": []
    })


# ─────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────
md([
    "# Module 2: Risk Metrics\n",
    "\n",
    "This notebook covers the risk measures that every portfolio manager, risk officer,\n",
    "and allocator cares about: **VaR, CVaR, drawdowns, Sortino, and Calmar**.\n",
    "\n",
    "By the end you will be able to:\n",
    "1. Compute Value at Risk three different ways and explain when each is appropriate\n",
    "2. Explain why CVaR (Expected Shortfall) replaced VaR in Basel III\n",
    "3. Analyse drawdown episodes and identify the worst peak-to-trough periods\n",
    "4. Calculate Sortino and Calmar ratios and explain why they can be more informative than Sharpe\n",
    "\n",
    "**How to use:** Run each cell with Shift+Enter. Read the explanations, then change\n",
    "the ticker or confidence level to build intuition.\n",
    "\n",
    "---"
])

# ─────────────────────────────────────────────────────────
# SETUP CELL
# ─────────────────────────────────────────────────────────
code([
    "# Setup — run this first\n",
    "import sys, os\n",
    "sys.path.insert(0, os.path.abspath(\"..\"))\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy import stats\n",
    "from data_loader import fetch_prices\n",
    "\n",
    "%matplotlib inline\n",
    "plt.rcParams[\"figure.figsize\"] = (12, 5)\n",
    "plt.rcParams[\"axes.grid\"] = True\n",
    "plt.rcParams[\"grid.alpha\"] = 0.3\n",
    "TRADING_DAYS = 252\n",
    "print(\"Ready.\")"
])

# ─────────────────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────────────────
md([
    "## 0. Fetch Data & Compute Returns\n",
    "\n",
    "We will use **SPY** as the running example. Change `TICKER` below to explore any asset."
])

code([
    "# ============================\n",
    "# CHANGE THESE\n",
    "# ============================\n",
    "TICKER = \"SPY\"\n",
    "START  = \"2018-01-01\"\n",
    "END    = \"2025-12-31\"\n",
    "CONFIDENCE = 0.95          # for VaR / CVaR\n",
    "RISK_FREE  = 0.04          # annualised risk-free rate\n",
    "\n",
    "prices  = fetch_prices(TICKER, START, END)\n",
    "returns = prices.pct_change().dropna()\n",
    "\n",
    "print(f\"Loaded {len(prices)} days of {TICKER}\")\n",
    "print(f\"Date range: {prices.index[0].date()} to {prices.index[-1].date()}\")\n",
    "print(f\"Confidence level: {CONFIDENCE:.0%}\")\n",
    "print(f\"Risk-free rate:   {RISK_FREE:.2%} annualised\")"
])

# ═════════════════════════════════════════════════════════
# SECTION 1 — VALUE AT RISK
# ═════════════════════════════════════════════════════════
md([
    "---\n",
    "## 1. Value at Risk (VaR)\n",
    "\n",
    "**VaR answers: what is the worst loss I can expect X% of the time?**\n",
    "\n",
    "More precisely, the $\\alpha$-level VaR is the threshold such that the probability\n",
    "of a loss exceeding that threshold is at most $(1 - \\alpha)$.\n",
    "\n",
    "For example, a 95% daily VaR of $-1.5\\%$ means:\n",
    "> \"On 95% of trading days, the portfolio will not lose more than 1.5%.\n",
    "> On the remaining 5% of days, losses may exceed 1.5%.\"\n",
    "\n",
    "We will compute VaR three ways:\n",
    "\n",
    "### 1a. Historical VaR\n",
    "\n",
    "No distributional assumptions. Simply take the empirical quantile:\n",
    "\n",
    "$$\\text{VaR}_{\\alpha}^{\\text{hist}} = \\text{quantile of returns at } (1 - \\alpha) \\text{ level}$$\n",
    "\n",
    "For 95% confidence, this is the 5th percentile of the return distribution.\n",
    "\n",
    "### 1b. Parametric (Gaussian) VaR\n",
    "\n",
    "Assume returns are normally distributed with mean $\\mu$ and standard deviation $\\sigma$:\n",
    "\n",
    "$$VaR_{\\alpha} = \\mu + z_{\\alpha} \\cdot \\sigma$$\n",
    "\n",
    "where $z_{\\alpha}$ is the normal quantile at level $(1 - \\alpha)$. For 95% confidence,\n",
    "$z_{0.95} \\approx -1.645$.\n",
    "\n",
    "**Problem:** this assumes normality. We saw in Module 1 that returns have fat tails,\n",
    "so parametric VaR *underestimates* true tail risk.\n",
    "\n",
    "### 1c. Monte Carlo VaR (Student-t)\n",
    "\n",
    "Fit a Student-t distribution to the returns (which captures fat tails), then\n",
    "simulate a large number of returns and take the quantile:\n",
    "\n",
    "1. Estimate parameters $(\\nu, \\mu, \\sigma)$ of a Student-t distribution via MLE\n",
    "2. Simulate $N = 100{,}000$ returns from $t(\\nu, \\mu, \\sigma)$\n",
    "3. $VaR_{\\alpha}^{MC} = \\text{quantile of simulated returns at } (1-\\alpha)$\n",
    "\n",
    "The Student-t distribution has a degrees-of-freedom parameter $\\nu$ that controls\n",
    "tail thickness. Lower $\\nu$ = fatter tails. ($\\nu \\to \\infty$ recovers the normal.)"
])

code([
    "# ── 1a. Historical VaR ──────────────────────────────────\n",
    "alpha = 1 - CONFIDENCE\n",
    "var_hist = returns.quantile(alpha)\n",
    "\n",
    "# ── 1b. Parametric (Gaussian) VaR ──────────────────────\n",
    "mu  = returns.mean()\n",
    "sig = returns.std()\n",
    "z_alpha = stats.norm.ppf(alpha)\n",
    "var_param = mu + z_alpha * sig\n",
    "\n",
    "# ── 1c. Monte Carlo VaR (Student-t) ───────────────────\n",
    "# Fit Student-t to the returns\n",
    "nu, t_mu, t_sig = stats.t.fit(returns)\n",
    "print(f\"Student-t fit: df={nu:.2f}, loc={t_mu:.6f}, scale={t_sig:.6f}\")\n",
    "\n",
    "np.random.seed(42)\n",
    "n_sims = 100_000\n",
    "sim_returns = stats.t.rvs(df=nu, loc=t_mu, scale=t_sig, size=n_sims)\n",
    "var_mc = np.percentile(sim_returns, alpha * 100)\n",
    "\n",
    "# ── Compare ───────────────────────────────────────────\n",
    "var_table = pd.DataFrame({\n",
    "    \"Method\": [\"Historical\", \"Parametric (Normal)\", \"Monte Carlo (Student-t)\"],\n",
    "    f\"{CONFIDENCE:.0%} VaR (daily)\": [f\"{var_hist:.4%}\", f\"{var_param:.4%}\", f\"{var_mc:.4%}\"],\n",
    "})\n",
    "print(f\"\\n{CONFIDENCE:.0%} Value at Risk — {TICKER}\")\n",
    "print(var_table.to_string(index=False))\n",
    "print()\n",
    "print(\"Interpretation:\")\n",
    "print(f\"  On {CONFIDENCE:.0%} of days, {TICKER} should not lose more than ~{abs(var_hist):.2%}.\")\n",
    "print(f\"  The parametric VaR {'underestimates' if abs(var_param) < abs(var_hist) else 'overestimates'} risk vs historical.\")\n",
    "print(f\"  Monte Carlo with Student-t typically captures fat tails better than the Normal.\")"
])

# ═════════════════════════════════════════════════════════
# SECTION 2 — CVaR / EXPECTED SHORTFALL
# ═════════════════════════════════════════════════════════
md([
    "---\n",
    "## 2. Conditional VaR (CVaR / Expected Shortfall)\n",
    "\n",
    "VaR tells you the *threshold* — but what happens when you cross it?\n",
    "CVaR (also called Expected Shortfall) answers that question.\n",
    "\n",
    "$$CVaR_{\\alpha} = E[r \\mid r \\leq VaR_{\\alpha}]$$\n",
    "\n",
    "This is the **average loss in the worst $(1-\\alpha)$ of days** — the mean of everything\n",
    "beyond the VaR cutoff.\n",
    "\n",
    "### Why CVaR is better than VaR\n",
    "\n",
    "| Property | VaR | CVaR |\n",
    "|----------|-----|------|\n",
    "| Tells you the threshold | Yes | Yes (it includes VaR) |\n",
    "| Tells you *how bad* the tail is | No | **Yes** |\n",
    "| Coherent risk measure | **No** | **Yes** |\n",
    "| Sub-additive (diversification helps) | Not always | Always |\n",
    "| Used in Basel III/IV | Replaced | **Current standard** |\n",
    "\n",
    "**Coherence** matters because a non-coherent measure can say that combining two portfolios\n",
    "is *riskier* than holding them separately — which contradicts the whole point of diversification.\n",
    "\n",
    "**Basel III moved from VaR to CVaR** precisely because CVaR captures tail severity.\n",
    "A bank could have the same VaR but wildly different tail behaviour; CVaR distinguishes them."
])

code([
    "# ── Historical CVaR ────────────────────────────────────\n",
    "tail_returns = returns[returns <= var_hist]\n",
    "cvar_hist = tail_returns.mean()\n",
    "\n",
    "# ── Parametric CVaR (Normal) ──────────────────────────\n",
    "cvar_param = mu - sig * stats.norm.pdf(z_alpha) / alpha\n",
    "\n",
    "# ── Monte Carlo CVaR (Student-t) ─────────────────────\n",
    "cvar_mc = sim_returns[sim_returns <= var_mc].mean()\n",
    "\n",
    "cvar_table = pd.DataFrame({\n",
    "    \"Method\": [\"Historical\", \"Parametric (Normal)\", \"Monte Carlo (Student-t)\"],\n",
    "    f\"{CONFIDENCE:.0%} VaR\": [f\"{var_hist:.4%}\", f\"{var_param:.4%}\", f\"{var_mc:.4%}\"],\n",
    "    f\"{CONFIDENCE:.0%} CVaR\": [f\"{cvar_hist:.4%}\", f\"{cvar_param:.4%}\", f\"{cvar_mc:.4%}\"],\n",
    "})\n",
    "print(f\"{CONFIDENCE:.0%} VaR vs CVaR — {TICKER}\")\n",
    "print(cvar_table.to_string(index=False))\n",
    "print()\n",
    "print(f\"CVaR is always worse (more negative) than VaR.\")\n",
    "print(f\"The gap tells you how severe the tail is: CVaR/VaR = {cvar_hist/var_hist:.2f}\")"
])

md([
    "### Visualising VaR & CVaR on the return distribution"
])

code([
    "fig, ax = plt.subplots(figsize=(12, 5))\n",
    "\n",
    "# Histogram of returns\n",
    "n, bins, patches = ax.hist(returns, bins=100, density=True, alpha=0.6,\n",
    "                           color=\"steelblue\", edgecolor=\"none\",\n",
    "                           label=\"Return distribution\")\n",
    "\n",
    "# Shade the tail (below VaR)\n",
    "for i, (b_left, b_right) in enumerate(zip(bins[:-1], bins[1:])):\n",
    "    if b_right <= var_hist:\n",
    "        patches[i].set_facecolor(\"crimson\")\n",
    "        patches[i].set_alpha(0.8)\n",
    "\n",
    "# VaR line\n",
    "ax.axvline(var_hist, color=\"red\", linewidth=2, linestyle=\"--\",\n",
    "           label=f\"VaR {CONFIDENCE:.0%} = {var_hist:.2%}\")\n",
    "\n",
    "# CVaR line\n",
    "ax.axvline(cvar_hist, color=\"darkred\", linewidth=2, linestyle=\"-\",\n",
    "           label=f\"CVaR {CONFIDENCE:.0%} = {cvar_hist:.2%}\")\n",
    "\n",
    "ax.set_title(f\"{TICKER} — VaR & CVaR at {CONFIDENCE:.0%} Confidence\", fontsize=14)\n",
    "ax.set_xlabel(\"Daily Return\")\n",
    "ax.set_ylabel(\"Density\")\n",
    "ax.legend(fontsize=11)\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"../module2_SPY_var_cvar.png\", dpi=150, bbox_inches=\"tight\")\n",
    "plt.show()\n",
    "\n",
    "print(\"Red dashed line = VaR (threshold for the worst 5% of days)\")\n",
    "print(\"Dark red solid line = CVaR (average loss in those worst 5% days)\")\n",
    "print(\"Shaded red area = the tail that CVaR averages over\")"
])

# ═════════════════════════════════════════════════════════
# SECTION 3 — DRAWDOWN ANALYSIS
# ═════════════════════════════════════════════════════════
md([
    "---\n",
    "## 3. Drawdown Analysis\n",
    "\n",
    "A **drawdown** measures the decline from a historical peak. It captures the pain\n",
    "an investor actually feels — \"I was up 30%, now I'm only up 10%.\"\n",
    "\n",
    "### Formula\n",
    "\n",
    "Let $P_t^{\\max} = \\max_{s \\leq t} P_s$ be the running maximum price up to time $t$.\n",
    "\n",
    "$$DD_t = \\frac{P_t - P_t^{\\max}}{P_t^{\\max}}$$\n",
    "\n",
    "The drawdown is always $\\leq 0$ (you are either at a peak or below one).\n",
    "\n",
    "### Maximum Drawdown\n",
    "\n",
    "$$\\text{MaxDrawdown} = \\min_t DD_t$$\n",
    "\n",
    "This is the single worst peak-to-trough decline over the period.\n",
    "\n",
    "### Why LPs care\n",
    "\n",
    "- Volatility is symmetric — it treats upside and downside the same\n",
    "- Drawdown captures the **actual pain** of losses\n",
    "- A fund with 20% annual vol and -50% max drawdown is VERY different from one\n",
    "  with 20% vol and -15% max drawdown\n",
    "- Many fund mandates have hard drawdown limits (e.g., \"liquidate at -20%\")"
])

code([
    "# ── Compute drawdown series ────────────────────────────\n",
    "cumulative  = (1 + returns).cumprod()\n",
    "running_max = cumulative.cummax()\n",
    "drawdown    = (cumulative - running_max) / running_max\n",
    "\n",
    "max_dd = drawdown.min()\n",
    "max_dd_date = drawdown.idxmin()\n",
    "\n",
    "# Find the peak before the max drawdown\n",
    "peak_date = cumulative.loc[:max_dd_date].idxmax()\n",
    "\n",
    "print(f\"Maximum Drawdown: {max_dd:.2%}\")\n",
    "print(f\"Peak date:        {peak_date.date()}\")\n",
    "print(f\"Trough date:      {max_dd_date.date()}\")\n",
    "print(f\"Duration:         {(max_dd_date - peak_date).days} calendar days\")"
])

code([
    "# ── Plot price with drawdown shading ───────────────────\n",
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,\n",
    "                                gridspec_kw={\"height_ratios\": [2, 1]})\n",
    "\n",
    "# Top: cumulative return (normalised to 1)\n",
    "ax1.plot(cumulative, color=\"steelblue\", linewidth=0.9, label=\"Cumulative Return\")\n",
    "ax1.plot(running_max, color=\"grey\", linewidth=0.5, linestyle=\"--\", alpha=0.7,\n",
    "         label=\"Running Max\")\n",
    "ax1.set_title(f\"{TICKER} — Cumulative Return & Drawdowns\", fontsize=14)\n",
    "ax1.set_ylabel(\"Growth of $1\")\n",
    "ax1.legend()\n",
    "\n",
    "# Bottom: drawdown chart\n",
    "ax2.fill_between(drawdown.index, drawdown.values, 0,\n",
    "                 color=\"crimson\", alpha=0.4)\n",
    "ax2.plot(drawdown, color=\"crimson\", linewidth=0.5)\n",
    "ax2.set_ylabel(\"Drawdown\")\n",
    "ax2.set_title(\"Drawdown from Peak\")\n",
    "\n",
    "# Mark the maximum drawdown\n",
    "ax2.annotate(f\"Max DD: {max_dd:.1%}\",\n",
    "             xy=(max_dd_date, max_dd),\n",
    "             xytext=(max_dd_date, max_dd - 0.05),\n",
    "             fontsize=10, fontweight=\"bold\", color=\"darkred\",\n",
    "             arrowprops=dict(arrowstyle=\"->\", color=\"darkred\"))\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"../module2_SPY_drawdowns.png\", dpi=150, bbox_inches=\"tight\")\n",
    "plt.show()"
])

md([
    "### Top 5 drawdown episodes\n",
    "\n",
    "Let us identify the largest distinct drawdown episodes by finding periods where\n",
    "the drawdown dips below a threshold and then recovers."
])

code([
    "def find_drawdown_episodes(dd_series, top_n=5):\n",
    "    \"\"\"\n",
    "    Identify the top N distinct drawdown episodes.\n",
    "    A new episode starts when drawdown returns to 0 after a trough.\n",
    "    \"\"\"\n",
    "    episodes = []\n",
    "    in_dd = False\n",
    "    start = None\n",
    "    worst = 0\n",
    "    worst_date = None\n",
    "\n",
    "    for date, val in dd_series.items():\n",
    "        if val < 0 and not in_dd:\n",
    "            in_dd = True\n",
    "            start = date\n",
    "            worst = val\n",
    "            worst_date = date\n",
    "        elif val < 0 and in_dd:\n",
    "            if val < worst:\n",
    "                worst = val\n",
    "                worst_date = date\n",
    "        elif val >= 0 and in_dd:\n",
    "            episodes.append({\n",
    "                \"Start\": start.date(),\n",
    "                \"Trough\": worst_date.date(),\n",
    "                \"Recovery\": date.date(),\n",
    "                \"Max DD\": worst,\n",
    "                \"Days to Trough\": (worst_date - start).days,\n",
    "                \"Days to Recover\": (date - worst_date).days,\n",
    "            })\n",
    "            in_dd = False\n",
    "            worst = 0\n",
    "\n",
    "    # If still in a drawdown at the end\n",
    "    if in_dd:\n",
    "        episodes.append({\n",
    "            \"Start\": start.date(),\n",
    "            \"Trough\": worst_date.date(),\n",
    "            \"Recovery\": \"ongoing\",\n",
    "            \"Max DD\": worst,\n",
    "            \"Days to Trough\": (worst_date - start).days,\n",
    "            \"Days to Recover\": \"ongoing\",\n",
    "        })\n",
    "\n",
    "    df = pd.DataFrame(episodes)\n",
    "    df = df.sort_values(\"Max DD\").head(top_n).reset_index(drop=True)\n",
    "    df.index = df.index + 1\n",
    "    df.index.name = \"Rank\"\n",
    "    return df\n",
    "\n",
    "top5 = find_drawdown_episodes(drawdown, top_n=5)\n",
    "\n",
    "# Format the Max DD column as percentage\n",
    "display_df = top5.copy()\n",
    "display_df[\"Max DD\"] = display_df[\"Max DD\"].apply(lambda x: f\"{x:.2%}\")\n",
    "\n",
    "print(f\"Top 5 Drawdown Episodes — {TICKER}\")\n",
    "print(display_df.to_string())\n",
    "print()\n",
    "print(\"Notice how the worst drawdowns often recover slower than they fell.\")\n",
    "print(\"This asymmetry is a hallmark of equity markets.\")"
])

# ═════════════════════════════════════════════════════════
# SECTION 4 — SORTINO RATIO
# ═════════════════════════════════════════════════════════
md([
    "---\n",
    "## 4. Sortino Ratio\n",
    "\n",
    "### The problem with Sharpe\n",
    "\n",
    "The Sharpe ratio penalises **all** volatility equally:\n",
    "\n",
    "$$Sharpe = \\frac{r_p - r_f}{\\sigma}$$\n",
    "\n",
    "But wait — **upside volatility is good!** If a fund makes +5% one day and +3%\n",
    "the next, that increases $\\sigma$ and *hurts* the Sharpe ratio. That makes no sense.\n",
    "\n",
    "### The Sortino fix\n",
    "\n",
    "The Sortino ratio only penalises **downside** volatility:\n",
    "\n",
    "$$Sortino = \\frac{r_p - r_f}{\\sigma_d}$$\n",
    "\n",
    "where the downside deviation is:\n",
    "\n",
    "$$\\sigma_d = \\sqrt{\\frac{1}{N}\\sum_{r_t < r_f}(r_t - r_f)^2}$$\n",
    "\n",
    "Note: we sum over **all** observations $N$, but only the terms where $r_t < r_f$\n",
    "contribute non-zero values. This correctly reduces $\\sigma_d$ when there are\n",
    "fewer downside observations.\n",
    "\n",
    "### Interpretation\n",
    "\n",
    "- **Sortino > Sharpe** means the asset has more upside volatility than downside.\n",
    "  This is the *good* kind of volatility — asymmetric returns in your favour.\n",
    "- **Sortino < Sharpe** means downside volatility dominates. Beware.\n",
    "- **\"Sharpe penalises ALL vol. Sortino only penalises DOWNSIDE vol. Upside vol is good!\"**"
])

code([
    "# ── Sharpe Ratio ───────────────────────────────────────\n",
    "rf_daily = RISK_FREE / TRADING_DAYS\n",
    "excess_returns = returns - rf_daily\n",
    "\n",
    "sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(TRADING_DAYS)\n",
    "\n",
    "# ── Sortino Ratio ─────────────────────────────────────\n",
    "downside = excess_returns.copy()\n",
    "downside[downside > 0] = 0                          # zero out positive days\n",
    "downside_std = np.sqrt((downside ** 2).mean())       # downside deviation (daily)\n",
    "sortino = excess_returns.mean() / downside_std * np.sqrt(TRADING_DAYS)\n",
    "\n",
    "print(f\"Sharpe Ratio:   {sharpe:.3f}\")\n",
    "print(f\"Sortino Ratio:  {sortino:.3f}\")\n",
    "print()\n",
    "\n",
    "# Decompose volatility\n",
    "total_vol = returns.std() * np.sqrt(TRADING_DAYS)\n",
    "downside_vol = downside_std * np.sqrt(TRADING_DAYS)\n",
    "upside = returns.copy()\n",
    "upside[upside < rf_daily] = 0\n",
    "upside_vol = np.sqrt((upside ** 2).mean()) * np.sqrt(TRADING_DAYS)\n",
    "\n",
    "print(f\"Total annualised vol:    {total_vol:.2%}\")\n",
    "print(f\"Downside annualised vol: {downside_vol:.2%}\")\n",
    "print(f\"Upside annualised vol:   {upside_vol:.2%}\")\n",
    "print()\n",
    "\n",
    "if sortino > sharpe:\n",
    "    print(f\"Sortino ({sortino:.2f}) > Sharpe ({sharpe:.2f})\")\n",
    "    print(\"  -> More upside vol than downside vol. The return distribution is favourably skewed.\")\n",
    "else:\n",
    "    print(f\"Sortino ({sortino:.2f}) <= Sharpe ({sharpe:.2f})\")\n",
    "    print(\"  -> Downside vol dominates. The return distribution has a heavy left tail.\")"
])

code([
    "# ── Visualise upside vs downside returns ──────────────\n",
    "fig, ax = plt.subplots(figsize=(12, 5))\n",
    "\n",
    "up_days   = returns[returns >= rf_daily]\n",
    "down_days = returns[returns < rf_daily]\n",
    "\n",
    "ax.hist(up_days, bins=60, alpha=0.6, color=\"seagreen\", label=f\"Upside days ({len(up_days)})\")\n",
    "ax.hist(down_days, bins=60, alpha=0.6, color=\"crimson\", label=f\"Downside days ({len(down_days)})\")\n",
    "ax.axvline(rf_daily, color=\"black\", linestyle=\"--\", linewidth=1,\n",
    "           label=f\"Rf = {rf_daily:.4%}/day\")\n",
    "ax.set_title(f\"{TICKER} — Upside vs Downside Returns (Sortino decomposition)\", fontsize=13)\n",
    "ax.set_xlabel(\"Daily Return\")\n",
    "ax.set_ylabel(\"Frequency\")\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"Sortino only cares about the red bars. The green bars are free.\")"
])

# ═════════════════════════════════════════════════════════
# SECTION 5 — CALMAR RATIO
# ═════════════════════════════════════════════════════════
md([
    "---\n",
    "## 5. Calmar Ratio\n",
    "\n",
    "The Calmar ratio measures **return per unit of maximum pain**:\n",
    "\n",
    "$$Calmar = \\frac{r_{\\text{annual}} - r_f}{|\\text{MaxDrawdown}|}$$\n",
    "\n",
    "### Interpretation\n",
    "\n",
    "- **Calmar = 1.0** means you earned 1% annualised for every 1% of max drawdown suffered\n",
    "- **Calmar > 1.0** is generally considered good — the return more than compensates for the pain\n",
    "- **Calmar < 0.5** is poor — you suffered a lot of drawdown for the return you got\n",
    "\n",
    "### Why allocators love it\n",
    "\n",
    "LPs (limited partners) care about drawdowns **more** than volatility because:\n",
    "1. Drawdowns trigger redemptions (investor psychology)\n",
    "2. Drawdowns can breach fund mandates and force liquidation\n",
    "3. Recovery from drawdowns is non-linear: a -50% drawdown requires a +100% gain to recover\n",
    "\n",
    "A fund with a Sharpe of 1.5 but a Calmar of 0.3 had a catastrophic drawdown\n",
    "that the Sharpe ratio hides. The Calmar ratio exposes it."
])

code([
    "# ── Calmar Ratio ───────────────────────────────────────\n",
    "annual_return = (cumulative.iloc[-1]) ** (TRADING_DAYS / len(returns)) - 1\n",
    "calmar = (annual_return - RISK_FREE) / abs(max_dd)\n",
    "\n",
    "print(f\"Annualised return:  {annual_return:.2%}\")\n",
    "print(f\"Risk-free rate:     {RISK_FREE:.2%}\")\n",
    "print(f\"Max drawdown:       {max_dd:.2%}\")\n",
    "print(f\"Calmar ratio:       {calmar:.3f}\")\n",
    "print()\n",
    "print(f\"Interpretation: for every 1% of max drawdown pain,\")\n",
    "print(f\"  {TICKER} earned {calmar:.2f}% of excess annual return.\")\n",
    "print()\n",
    "recovery_needed = 1 / (1 + max_dd) - 1\n",
    "print(f\"Note: a {max_dd:.1%} drawdown requires a {recovery_needed:.1%} gain to recover.\")\n",
    "print(f\"This non-linearity is why drawdowns are so painful.\")"
])

# ═════════════════════════════════════════════════════════
# SECTION 6 — COMPARISON DASHBOARD
# ═════════════════════════════════════════════════════════
md([
    "---\n",
    "## 6. Risk Metrics Dashboard\n",
    "\n",
    "Let us bring all the metrics together in a single summary table for **SPY**."
])

code([
    "# ── All metrics side by side ───────────────────────────\n",
    "dashboard = pd.DataFrame({\n",
    "    \"Metric\": [\n",
    "        \"Annual Return\",\n",
    "        \"Annual Volatility\",\n",
    "        f\"VaR {CONFIDENCE:.0%} (daily, historical)\",\n",
    "        f\"CVaR {CONFIDENCE:.0%} (daily, historical)\",\n",
    "        \"Maximum Drawdown\",\n",
    "        \"Sharpe Ratio\",\n",
    "        \"Sortino Ratio\",\n",
    "        \"Calmar Ratio\",\n",
    "    ],\n",
    "    \"Value\": [\n",
    "        f\"{annual_return:.2%}\",\n",
    "        f\"{total_vol:.2%}\",\n",
    "        f\"{var_hist:.4%}\",\n",
    "        f\"{cvar_hist:.4%}\",\n",
    "        f\"{max_dd:.2%}\",\n",
    "        f\"{sharpe:.3f}\",\n",
    "        f\"{sortino:.3f}\",\n",
    "        f\"{calmar:.3f}\",\n",
    "    ]\n",
    "})\n",
    "\n",
    "print(f\"Risk Metrics Dashboard — {TICKER}\")\n",
    "print(f\"{'=' * 50}\")\n",
    "print(dashboard.to_string(index=False))\n",
    "print(f\"{'=' * 50}\")\n",
    "print()\n",
    "print(\"This is essentially a one-page risk report for the asset.\")\n",
    "print(\"At a fund, you would see this for every book, every strategy, every day.\")"
])

code([
    "# ── Visual dashboard ───────────────────────────────────\n",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
    "\n",
    "# (1) Cumulative returns\n",
    "ax = axes[0, 0]\n",
    "ax.plot(cumulative, color=\"steelblue\", linewidth=0.9)\n",
    "ax.set_title(f\"{TICKER} Cumulative Return\")\n",
    "ax.set_ylabel(\"Growth of $1\")\n",
    "\n",
    "# (2) Return distribution with VaR/CVaR\n",
    "ax = axes[0, 1]\n",
    "ax.hist(returns, bins=80, density=True, alpha=0.6, color=\"steelblue\", edgecolor=\"none\")\n",
    "ax.axvline(var_hist, color=\"red\", ls=\"--\", lw=1.5, label=f\"VaR = {var_hist:.2%}\")\n",
    "ax.axvline(cvar_hist, color=\"darkred\", ls=\"-\", lw=1.5, label=f\"CVaR = {cvar_hist:.2%}\")\n",
    "ax.set_title(f\"Return Distribution + VaR/CVaR\")\n",
    "ax.legend(fontsize=9)\n",
    "\n",
    "# (3) Drawdown chart\n",
    "ax = axes[1, 0]\n",
    "ax.fill_between(drawdown.index, drawdown.values, 0, color=\"crimson\", alpha=0.4)\n",
    "ax.plot(drawdown, color=\"crimson\", linewidth=0.5)\n",
    "ax.set_title(\"Drawdown from Peak\")\n",
    "ax.set_ylabel(\"Drawdown\")\n",
    "\n",
    "# (4) Ratios bar chart\n",
    "ax = axes[1, 1]\n",
    "ratios = {\"Sharpe\": sharpe, \"Sortino\": sortino, \"Calmar\": calmar}\n",
    "colours = [\"steelblue\", \"seagreen\", \"darkorange\"]\n",
    "bars = ax.bar(ratios.keys(), ratios.values(), color=colours, alpha=0.8, edgecolor=\"grey\")\n",
    "for bar, val in zip(bars, ratios.values()):\n",
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,\n",
    "            f\"{val:.2f}\", ha=\"center\", fontsize=11, fontweight=\"bold\")\n",
    "ax.set_title(\"Risk-Adjusted Return Ratios\")\n",
    "ax.set_ylabel(\"Ratio\")\n",
    "\n",
    "plt.suptitle(f\"{TICKER} — Risk Metrics Dashboard\", fontsize=15, fontweight=\"bold\", y=1.01)\n",
    "plt.tight_layout()\n",
    "plt.show()"
])

# ═════════════════════════════════════════════════════════
# EXERCISES
# ═════════════════════════════════════════════════════════
md([
    "---\n",
    "## Exercises\n",
    "\n",
    "### Exercise 1: Different confidence levels\n",
    "Change `CONFIDENCE` to 0.99 at the top and re-run the notebook.\n",
    "- How much worse does VaR get at 99% vs 95%?\n",
    "- How much worse does CVaR get?\n",
    "- Which method (Historical, Parametric, Monte Carlo) shows the biggest difference?\n",
    "\n",
    "### Exercise 2: Compare assets\n",
    "Re-run with `TICKER = \"QQQ\"` (Nasdaq) and then `TICKER = \"TLT\"` (long-term bonds).\n",
    "- Which has higher VaR? Which has higher Sortino?\n",
    "- Does TLT have positive or negative skew? Why?\n",
    "\n",
    "### Exercise 3: Rolling VaR\n",
    "Compute a 60-day rolling historical VaR and plot it. Does VaR spike during crises?\n",
    "\n",
    "```python\n",
    "# Hint:\n",
    "rolling_var = returns.rolling(60).quantile(0.05)\n",
    "rolling_var.plot(title=\"60-Day Rolling 95% VaR\")\n",
    "```\n",
    "\n",
    "### Exercise 4: Drawdown recovery time\n",
    "Using the `find_drawdown_episodes` function, which drawdown episode took the longest\n",
    "to recover? Is recovery time correlated with drawdown depth?\n",
    "\n",
    "### Exercise 5: Sortino vs Sharpe scatter\n",
    "Compute Sharpe and Sortino for `[\"SPY\", \"QQQ\", \"TLT\", \"GLD\", \"XLE\"]`.\n",
    "Plot Sortino vs Sharpe. Assets above the 45-degree line have favourable skew."
])

code([
    "# Exercise 3: Rolling VaR (starter code)\n",
    "rolling_var = returns.rolling(60).quantile(0.05)\n",
    "rolling_cvar = returns.rolling(60).apply(lambda x: x[x <= x.quantile(0.05)].mean())\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 5))\n",
    "ax.plot(rolling_var, color=\"red\", linewidth=0.8, label=\"Rolling 95% VaR\")\n",
    "ax.plot(rolling_cvar, color=\"darkred\", linewidth=0.8, label=\"Rolling 95% CVaR\")\n",
    "ax.axhline(var_hist, color=\"red\", ls=\"--\", alpha=0.3, label=f\"Full-period VaR ({var_hist:.2%})\")\n",
    "ax.set_title(f\"{TICKER} — 60-Day Rolling VaR & CVaR\", fontsize=13)\n",
    "ax.set_ylabel(\"Daily Return\")\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"VaR and CVaR spike during volatile periods (COVID, 2022 rate hikes).\")\n",
    "print(\"A risk manager monitors rolling VaR daily to catch regime changes early.\")"
])

md([
    "---\n",
    "\n",
    "**Next up:** Module 3 covers factor models and regression analysis (CAPM, beta, alpha)."
])

# ─────────────────────────────────────────────────────────
# BUILD THE NOTEBOOK
# ─────────────────────────────────────────────────────────
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": cells
}

out_path = "/home/user/sharpe-guesser/notebooks/02_risk_metrics.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Wrote {len(cells)} cells to {out_path}")
