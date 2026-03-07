"""
Module 2: Risk Metrics — VaR, CVaR, Drawdowns
===============================================

This is your bread and butter as a risk associate. You've seen these
numbers on every risk report — now you'll build them from scratch.

After this module you'll understand exactly what happens inside the
risk system when it spits out "1-day 95% VaR is -2.1%."

Topics:
    1. Value at Risk (VaR) — historical, parametric, and Monte Carlo
    2. Conditional VaR (CVaR / Expected Shortfall)
    3. Drawdown analysis — max drawdown, drawdown duration
    4. Sortino ratio — Sharpe but smarter about downside risk
    5. Calmar ratio — return vs. max drawdown

Run:
    python -m modules.module_2_risk_metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 1. VALUE AT RISK (VaR)
# ---------------------------------------------------------------------------
# VaR answers: "What is the worst loss I can expect X% of the time?"
#
# Three methods, each with trade-offs:
#   - Historical: simple, no distribution assumptions, but limited by history
#   - Parametric: assumes normal distribution (we know that's wrong, but it's fast)
#   - Monte Carlo: flexible, can handle any distribution, but computationally heavy
#
# At your fund, you probably use historical VaR for daily reporting and
# Monte Carlo for stress testing.

def var_historical(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR — just look at the actual percentile of returns.

    This is the simplest and most common method. No assumptions about
    the distribution shape. The downside: you can only capture risks
    that have already happened.
    """
    return daily_returns.quantile(1 - confidence)


def var_parametric(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric (Gaussian) VaR — assume returns are normally distributed.

    Fast and closed-form, but understates tail risk because real returns
    have fatter tails than a normal distribution (see Module 1).
    """
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    return stats.norm.ppf(1 - confidence, mu, sigma)


def var_monte_carlo(
    daily_returns: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 100_000,
) -> float:
    """
    Monte Carlo VaR — simulate returns from a fitted distribution.

    Here we use a Student-t distribution, which captures fat tails
    better than a normal. This is closer to what your risk system
    actually does.
    """
    # Fit a Student-t distribution to the returns
    params = stats.t.fit(daily_returns.dropna())
    # Simulate
    simulated = stats.t.rvs(*params, size=n_simulations)
    return np.percentile(simulated, (1 - confidence) * 100)


def compare_var_methods(daily_returns: pd.Series, name: str = ""):
    """Compare all three VaR methods side by side."""
    print(f"\n  VaR Comparison for {name}:")
    print(f"  {'Method':<20} {'95% VaR':>10} {'99% VaR':>10}")
    print(f"  {'-'*42}")
    for method_name, func in [("Historical", var_historical),
                               ("Parametric", var_parametric),
                               ("Monte Carlo (t)", var_monte_carlo)]:
        v95 = func(daily_returns, 0.95)
        v99 = func(daily_returns, 0.99)
        print(f"  {method_name:<20} {v95:>10.2%} {v99:>10.2%}")


# ---------------------------------------------------------------------------
# 2. CONDITIONAL VAR (CVaR / Expected Shortfall)
# ---------------------------------------------------------------------------
# VaR tells you the threshold. CVaR tells you the average loss BEYOND
# that threshold. It answers: "When things are bad, HOW bad on average?"
#
# CVaR is considered a better risk measure because:
#   - It's coherent (satisfies subadditivity — diversification always helps)
#   - It captures tail severity, not just tail probability
#   - Basel III/IV moved to Expected Shortfall for exactly this reason

def cvar(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall).

    The average loss on days that are worse than the VaR threshold.
    This is what regulators care about now.
    """
    var = var_historical(daily_returns, confidence)
    return daily_returns[daily_returns <= var].mean()


def plot_var_cvar(daily_returns: pd.Series, confidence: float = 0.95, name: str = ""):
    """Visualise VaR and CVaR on the return distribution."""
    var_val = var_historical(daily_returns, confidence)
    cvar_val = cvar(daily_returns, confidence)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(daily_returns, bins=150, density=True, alpha=0.6, color="steelblue")
    ax.axvline(var_val, color="orange", linewidth=2,
               label=f"{confidence:.0%} VaR: {var_val:.2%}")
    ax.axvline(cvar_val, color="red", linewidth=2, linestyle="--",
               label=f"{confidence:.0%} CVaR: {cvar_val:.2%}")

    # Shade the tail
    tail = daily_returns[daily_returns <= var_val]
    ax.hist(tail, bins=50, density=True, alpha=0.4, color="red")

    ax.set_title(f"{name} — VaR and CVaR ({confidence:.0%})")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.close()


# ---------------------------------------------------------------------------
# 3. DRAWDOWN ANALYSIS
# ---------------------------------------------------------------------------
# Drawdown = how far you've fallen from your peak.
# Max drawdown = the worst peak-to-trough decline.
#
# This is arguably the most intuitive risk metric: "How much could I lose
# from my high-water mark?" Every PM knows their fund's max drawdown.

def drawdown_series(prices: pd.Series) -> pd.Series:
    """
    Compute the drawdown at each point in time.

    Returns a series of negative values representing the percentage
    decline from the running maximum.
    """
    cummax = prices.cummax()
    return (prices - cummax) / cummax


def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown as a negative percentage."""
    return drawdown_series(prices).min()


def drawdown_details(prices: pd.Series) -> pd.DataFrame:
    """
    Find the top 5 drawdown episodes: when they started, when they
    bottomed, and how long recovery took.
    """
    dd = drawdown_series(prices)
    is_in_drawdown = dd < 0

    # Find drawdown episodes
    episodes = []
    in_dd = False
    start_idx = None

    for i in range(len(dd)):
        if dd.iloc[i] < 0 and not in_dd:
            in_dd = True
            start_idx = i
        elif dd.iloc[i] >= 0 and in_dd:
            in_dd = False
            episode_dd = dd.iloc[start_idx:i]
            trough_idx = episode_dd.idxmin()
            episodes.append({
                "start": dd.index[start_idx],
                "trough": trough_idx,
                "recovery": dd.index[i],
                "depth": episode_dd.min(),
                "duration_days": (dd.index[i] - dd.index[start_idx]).days,
            })

    if in_dd:  # still in drawdown at end
        episode_dd = dd.iloc[start_idx:]
        trough_idx = episode_dd.idxmin()
        episodes.append({
            "start": dd.index[start_idx],
            "trough": trough_idx,
            "recovery": None,
            "depth": episode_dd.min(),
            "duration_days": (dd.index[-1] - dd.index[start_idx]).days,
        })

    df = pd.DataFrame(episodes).sort_values("depth").head(5).reset_index(drop=True)
    return df


def plot_drawdowns(prices: pd.Series, name: str = ""):
    """Plot price with drawdown shading."""
    dd = drawdown_series(prices)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(prices, linewidth=0.8, color="steelblue")
    ax1.set_title(f"{name} — Price")
    ax1.set_ylabel("Price ($)")

    ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    ax2.plot(dd, linewidth=0.5, color="darkred")
    ax2.set_title(f"{name} — Drawdown (Max: {dd.min():.1%})")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    plt.close()


# ---------------------------------------------------------------------------
# 4. SORTINO RATIO
# ---------------------------------------------------------------------------
# The Sharpe ratio penalises ALL volatility equally. But upside vol is
# good — you want big positive days! The Sortino ratio only penalises
# downside deviation.
#
# Sortino = (annualised excess return) / (annualised downside deviation)

def downside_deviation(daily_returns: pd.Series, annual_rf: float = 0.05) -> float:
    """Annualised downside deviation — only counts negative excess returns."""
    daily_rf = annual_rf / TRADING_DAYS
    excess = daily_returns - daily_rf
    downside = excess[excess < 0]
    return downside.std() * np.sqrt(TRADING_DAYS)


def sortino_ratio(daily_returns: pd.Series, annual_rf: float = 0.05) -> float:
    """
    Sortino ratio — Sharpe's smarter cousin.

    Uses downside deviation instead of total standard deviation.
    A Sortino > Sharpe means the asset has more upside than downside vol.
    """
    daily_rf = annual_rf / TRADING_DAYS
    excess = daily_returns - daily_rf
    ann_excess = excess.mean() * TRADING_DAYS
    dd = downside_deviation(daily_returns, annual_rf)
    if dd < 1e-10 or np.isnan(dd):
        return 0.0
    return ann_excess / dd


# ---------------------------------------------------------------------------
# 5. CALMAR RATIO
# ---------------------------------------------------------------------------
# Calmar = annualised return / abs(max drawdown)
#
# This tells you: "For every unit of max pain, how much return do I get?"
# It's especially important for hedge funds because LPs care about
# drawdowns more than volatility.

def calmar_ratio(prices: pd.Series, annual_rf: float = 0.05) -> float:
    """
    Calmar ratio — return per unit of max drawdown.

    A Calmar of 2.0 means you earned 2x your worst drawdown per year.
    Anything above 1.0 is respectable for a hedge fund.
    """
    returns = prices.pct_change().dropna()
    daily_rf = annual_rf / TRADING_DAYS
    ann_excess = (returns.mean() - daily_rf) * TRADING_DAYS
    mdd = abs(max_drawdown(prices))
    if mdd < 1e-10:
        return 0.0
    return ann_excess / mdd


# ---------------------------------------------------------------------------
# FULL RISK REPORT
# ---------------------------------------------------------------------------

def risk_report(ticker: str, start: str, end: str, annual_rf: float = 0.05):
    """Generate a complete risk report for a single asset."""
    print(f"\n{'='*60}")
    print(f"  RISK REPORT: {ticker}  ({start} to {end})")
    print(f"{'='*60}")

    prices = yf.download(ticker, start=start, end=end, progress=False)["Close"].squeeze()
    returns = prices.pct_change().dropna()

    # Sharpe & variants
    daily_rf = annual_rf / TRADING_DAYS
    excess = returns - daily_rf
    ann_excess = excess.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sr = ann_excess / ann_vol if ann_vol > 0 else 0.0
    sortino = sortino_ratio(returns, annual_rf)
    calmar = calmar_ratio(prices, annual_rf)

    print(f"\n  Risk-Adjusted Returns:")
    print(f"    Sharpe ratio   : {sr:>8.2f}")
    print(f"    Sortino ratio  : {sortino:>8.2f}")
    print(f"    Calmar ratio   : {calmar:>8.2f}")

    # VaR & CVaR
    print(f"\n  Value at Risk (1-day):")
    for conf in [0.95, 0.99]:
        v = var_historical(returns, conf)
        cv = cvar(returns, conf)
        print(f"    {conf:.0%} VaR  : {v:>8.2%}    CVaR: {cv:>8.2%}")

    # Drawdowns
    mdd = max_drawdown(prices)
    print(f"\n  Max Drawdown: {mdd:.2%}")

    dd_table = drawdown_details(prices)
    print(f"\n  Top 5 Drawdown Episodes:")
    print(f"  {'#':<4} {'Start':<12} {'Trough':<12} {'Depth':>8} {'Days':>6}")
    print(f"  {'-'*46}")
    for _, row in dd_table.iterrows():
        start_str = row["start"].strftime("%Y-%m-%d") if pd.notna(row["start"]) else "N/A"
        trough_str = row["trough"].strftime("%Y-%m-%d") if pd.notna(row["trough"]) else "N/A"
        print(f"  {_ + 1:<4} {start_str:<12} {trough_str:<12} {row['depth']:>8.1%} {row['duration_days']:>6}")

    # Plots
    compare_var_methods(returns, ticker)
    plot_var_cvar(returns, 0.95, ticker)
    plot_drawdowns(prices, ticker)

    print()


# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  MODULE 2: Risk Metrics")
    print("=" * 60)

    risk_report("SPY", "2018-01-01", "2025-12-31")

    print("\n  EXERCISES:")
    print("  1. Compare VaR across methods — which is most conservative and why?")
    print("  2. Run this during a crisis period (2020-01 to 2020-06) vs calm")
    print("     period (2017-01 to 2017-12) — how do the numbers change?")
    print("  3. Which has a higher Sortino: SPY or QQQ? What does that tell you?")
    print("  4. Add a function that computes 'rolling VaR' — 60-day window.")
    print("     Plot it over time. When does VaR spike?")
    print()


if __name__ == "__main__":
    main()
