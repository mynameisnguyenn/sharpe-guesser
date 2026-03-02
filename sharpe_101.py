"""
Sharpe Ratio 101 — Learn Quant Finance by Building
====================================================

This script walks through the Sharpe ratio from first principles,
building up from raw price data to annualized risk-adjusted returns.

Run each section interactively (e.g., in a Jupyter notebook or just
read through the comments) to build your intuition.

Concepts covered:
    1. Fetching real market data
    2. Computing returns (simple & log)
    3. Excess returns over the risk-free rate
    4. Annualizing volatility and return
    5. The Sharpe ratio itself
    6. Rolling Sharpe — how it changes over time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import fetch_prices as _fetch_prices


# ---------------------------------------------------------------------------
# 1. FETCHING DATA
# ---------------------------------------------------------------------------
# As a risk associate you probably pull data from Bloomberg or an internal
# data lake. yfinance is the free equivalent — same idea, different pipe.
# We use data_loader which tries yfinance first, then falls back to sample data.

def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close prices for a single ticker."""
    return _fetch_prices(ticker, start, end)


# ---------------------------------------------------------------------------
# 2. COMPUTING RETURNS
# ---------------------------------------------------------------------------
# Two flavours you'll see everywhere:
#   - Simple return:  (P_t / P_{t-1}) - 1
#   - Log return:     ln(P_t / P_{t-1})
#
# Log returns are additive over time (handy for math); simple returns are
# additive across a portfolio (handy for PnL). Quants use both.

def simple_returns(prices: pd.Series) -> pd.Series:
    """Compute daily simple returns from a price series."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns from a price series."""
    return np.log(prices / prices.shift(1)).dropna()


# ---------------------------------------------------------------------------
# 3. EXCESS RETURNS
# ---------------------------------------------------------------------------
# The Sharpe ratio measures return *above* the risk-free rate per unit of
# risk. At your fund, the risk-free rate is probably pulled from Fed Funds
# or 3-month T-bills. Here we'll just parameterise it.

def excess_returns(returns: pd.Series, annual_rf: float = 0.05) -> pd.Series:
    """
    Subtract the daily risk-free rate from each return observation.

    Parameters
    ----------
    returns : pd.Series
        Daily simple or log returns.
    annual_rf : float
        Annualised risk-free rate (e.g., 0.05 for 5%).
    """
    daily_rf = annual_rf / 252  # 252 trading days per year
    return returns - daily_rf


# ---------------------------------------------------------------------------
# 4. ANNUALIZING
# ---------------------------------------------------------------------------
# Daily numbers are noisy. We annualise so that a Sharpe of 1.0 always
# means "1 unit of excess return per unit of vol, per year."
#
# Key insight: volatility scales with sqrt(time), return scales linearly.
# This is why high-frequency strategies can have astronomical Sharpes —
# they compound small edges many times.

TRADING_DAYS = 252


def annualized_return(daily_returns: pd.Series) -> float:
    """Annualise mean daily return."""
    return daily_returns.mean() * TRADING_DAYS


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualise daily standard deviation (the sqrt(T) rule)."""
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


# ---------------------------------------------------------------------------
# 5. THE SHARPE RATIO
# ---------------------------------------------------------------------------
# Sharpe = (annualised excess return) / (annualised volatility)
#
# Rules of thumb you already know from your seat:
#   < 0.5  — not great
#   0.5–1  — acceptable for a long-only strategy
#   1–2    — good for a hedge fund strategy
#   > 2    — either amazing or you have a bug (check your data!)
#   > 3    — almost certainly a bug, or very short track record

def sharpe_ratio(
    daily_returns: pd.Series,
    annual_rf: float = 0.05,
) -> float:
    """
    Compute the annualised Sharpe ratio.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily simple returns.
    annual_rf : float
        Annualised risk-free rate.

    Returns
    -------
    float
        The annualised Sharpe ratio.
    """
    excess = excess_returns(daily_returns, annual_rf)
    ann_ret = annualized_return(excess)
    ann_vol = annualized_volatility(excess)
    if ann_vol == 0:
        return 0.0
    return ann_ret / ann_vol


# ---------------------------------------------------------------------------
# 6. ROLLING SHARPE
# ---------------------------------------------------------------------------
# A single number hides a lot. Rolling Sharpe shows *when* a strategy or
# asset was generating good risk-adjusted returns. This is what your PMs
# look at when they say "the strategy stopped working in Q3."

def rolling_sharpe(
    daily_returns: pd.Series,
    window: int = 63,  # ~3 months
    annual_rf: float = 0.05,
) -> pd.Series:
    """
    Compute a rolling annualised Sharpe ratio.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily simple returns.
    window : int
        Rolling window in trading days.
    annual_rf : float
        Annualised risk-free rate.
    """
    daily_rf = annual_rf / TRADING_DAYS
    excess = daily_returns - daily_rf

    rolling_mean = excess.rolling(window).mean() * TRADING_DAYS
    rolling_std = excess.rolling(window).std() * np.sqrt(TRADING_DAYS)

    return (rolling_mean / rolling_std).dropna()


# ---------------------------------------------------------------------------
# PUTTING IT ALL TOGETHER
# ---------------------------------------------------------------------------

def analyse_ticker(ticker: str, start: str, end: str, annual_rf: float = 0.05):
    """Full Sharpe analysis with plots for a single ticker."""
    print(f"\n{'='*60}")
    print(f"  Sharpe Analysis: {ticker}  ({start} to {end})")
    print(f"{'='*60}\n")

    # Fetch & compute
    prices = fetch_prices(ticker, start, end)
    rets = simple_returns(prices)
    sr = sharpe_ratio(rets, annual_rf)
    ann_ret = annualized_return(excess_returns(rets, annual_rf))
    ann_vol = annualized_volatility(rets)

    # Print summary
    print(f"  Annualised excess return : {ann_ret:>8.2%}")
    print(f"  Annualised volatility    : {ann_vol:>8.2%}")
    print(f"  Sharpe ratio             : {sr:>8.2f}")
    print(f"  Risk-free rate (annual)  : {annual_rf:>8.2%}")
    print()

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(prices, linewidth=0.8)
    axes[0].set_title(f"{ticker} — Price")
    axes[0].set_ylabel("Price ($)")

    axes[1].plot(rets, linewidth=0.4, alpha=0.7)
    axes[1].set_title(f"{ticker} — Daily Returns")
    axes[1].set_ylabel("Return")
    axes[1].axhline(0, color="grey", linewidth=0.5)

    rs = rolling_sharpe(rets, window=63, annual_rf=annual_rf)
    axes[2].plot(rs, linewidth=0.8, color="darkorange")
    axes[2].axhline(0, color="grey", linewidth=0.5)
    axes[2].axhline(1, color="green", linewidth=0.5, linestyle="--", label="Sharpe = 1")
    axes[2].axhline(-1, color="red", linewidth=0.5, linestyle="--", label="Sharpe = -1")
    axes[2].set_title(f"{ticker} — 63-day Rolling Sharpe")
    axes[2].set_ylabel("Sharpe Ratio")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{ticker}_sharpe_analysis.png", dpi=150)
    plt.show()
    print(f"  Chart saved to {ticker}_sharpe_analysis.png\n")


# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Try it with a familiar name — change these to whatever you like
    analyse_ticker("SPY", start="2020-01-01", end="2025-12-31", annual_rf=0.05)

    # Compare two assets side by side
    for t in ["AAPL", "TLT"]:
        analyse_ticker(t, start="2020-01-01", end="2025-12-31", annual_rf=0.05)
