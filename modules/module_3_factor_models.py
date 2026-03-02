"""
Module 3: Factor Models & Regression
======================================

Factor models are the language of institutional investing. When your PM
says "we're long momentum and short value," they're speaking in factors.
When risk says "your beta to rates is too high," that's a factor exposure.

This module teaches you to decompose returns into systematic factors
and stock-specific alpha — the core of quantitative investing.

Topics:
    1. CAPM — the simplest factor model (just beta to market)
    2. Fama-French 3-factor model (market, size, value)
    3. Rolling beta — how factor exposure changes over time
    4. Alpha & residual analysis
    5. Information ratio — how good is your alpha?

Run:
    python -m modules.module_3_factor_models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from data_loader import fetch_prices, fetch_multi_prices

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------------

def fetch_returns(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily returns for a ticker."""
    prices = fetch_prices(ticker, start, end)
    return prices.pct_change().dropna()


def fetch_multi_returns(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download daily returns for multiple tickers."""
    prices = fetch_multi_prices(tickers, start, end)
    return prices.pct_change().dropna()


# ---------------------------------------------------------------------------
# 1. CAPM — CAPITAL ASSET PRICING MODEL
# ---------------------------------------------------------------------------
# The simplest factor model: R_stock - Rf = alpha + beta * (R_market - Rf)
#
#   beta > 1  → more volatile than the market (tech stocks)
#   beta < 1  → less volatile (utilities, consumer staples)
#   beta < 0  → moves opposite to market (very rare, some hedging instruments)
#   alpha > 0 → excess return not explained by market exposure (the holy grail)
#
# At your fund, every PM's book has a beta target. This is why.

def capm_regression(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    annual_rf: float = 0.05,
) -> dict:
    """
    Run a CAPM regression: R_stock - Rf = alpha + beta * (R_market - Rf)

    Returns alpha (annualised), beta, R-squared, and the full model.
    """
    daily_rf = annual_rf / TRADING_DAYS

    # Align dates
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ["stock", "market"]

    # Excess returns
    y = aligned["stock"] - daily_rf
    x = aligned["market"] - daily_rf
    x = sm.add_constant(x)  # add intercept (alpha)

    model = sm.OLS(y, x).fit()

    return {
        "alpha_daily": model.params.iloc[0],
        "alpha_annual": model.params.iloc[0] * TRADING_DAYS,
        "beta": model.params.iloc[1],
        "r_squared": model.rsquared,
        "alpha_tstat": model.tvalues.iloc[0],
        "beta_tstat": model.tvalues.iloc[1],
        "model": model,
    }


def print_capm(result: dict, stock_name: str = "", market_name: str = ""):
    """Pretty-print CAPM regression results."""
    print(f"\n  CAPM: {stock_name} vs {market_name}")
    print(f"  {'-'*45}")
    print(f"    Alpha (annual) : {result['alpha_annual']:>8.2%}  (t={result['alpha_tstat']:.2f})")
    print(f"    Beta           : {result['beta']:>8.3f}  (t={result['beta_tstat']:.2f})")
    print(f"    R-squared      : {result['r_squared']:>8.1%}")
    print(f"    {'Significant alpha' if abs(result['alpha_tstat']) > 2 else 'No significant alpha'}")


# ---------------------------------------------------------------------------
# 2. FAMA-FRENCH 3-FACTOR MODEL
# ---------------------------------------------------------------------------
# CAPM says market exposure explains everything. Fama & French showed
# that two more factors matter:
#   - SMB (Small Minus Big): small-cap stocks outperform large-caps
#   - HML (High Minus Low): value stocks outperform growth stocks
#
# R - Rf = alpha + b1*(Mkt-Rf) + b2*SMB + b3*HML
#
# In practice, we'll construct simple proxies using ETFs since the actual
# Fama-French data requires a separate download.

def build_factor_proxies(start: str, end: str) -> pd.DataFrame:
    """
    Build simple factor proxies using ETFs.

    MKT = SPY (market)
    SMB = IWM - SPY (small minus large, approximate)
    HML = IWD - IWF (value minus growth)

    These are rough proxies. Real factor data comes from Ken French's
    data library, but ETFs give you the intuition.
    """
    tickers = ["SPY", "IWM", "IWD", "IWF"]
    prices = fetch_multi_prices(tickers, start, end)
    rets = prices.pct_change().dropna()

    factors = pd.DataFrame(index=rets.index)
    factors["MKT"] = rets["SPY"]
    factors["SMB"] = rets["IWM"] - rets["SPY"]   # small minus big
    factors["HML"] = rets["IWD"] - rets["IWF"]   # value minus growth

    return factors


def multifactor_regression(
    stock_returns: pd.Series,
    factors: pd.DataFrame,
    annual_rf: float = 0.05,
) -> dict:
    """
    Run a multi-factor regression.

    R_stock - Rf = alpha + b1*MKT + b2*SMB + b3*HML + epsilon
    """
    daily_rf = annual_rf / TRADING_DAYS

    # Align dates
    aligned = pd.concat([stock_returns, factors], axis=1).dropna()
    y = aligned.iloc[:, 0] - daily_rf
    x = aligned.iloc[:, 1:]
    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()

    result = {
        "alpha_daily": model.params.iloc[0],
        "alpha_annual": model.params.iloc[0] * TRADING_DAYS,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "model": model,
    }

    # Extract factor loadings
    for i, name in enumerate(factors.columns):
        result[f"beta_{name}"] = model.params.iloc[i + 1]
        result[f"tstat_{name}"] = model.tvalues.iloc[i + 1]

    return result


def print_multifactor(result: dict, name: str = ""):
    """Pretty-print multi-factor regression results."""
    print(f"\n  Multi-Factor Model: {name}")
    print(f"  {'-'*50}")
    print(f"    Alpha (annual) : {result['alpha_annual']:>8.2%}")
    print(f"    R-squared      : {result['r_squared']:>8.1%}")
    print(f"    Adj R-squared  : {result['adj_r_squared']:>8.1%}")
    print(f"  Factor Loadings:")
    for key in result:
        if key.startswith("beta_"):
            factor = key.replace("beta_", "")
            tstat = result.get(f"tstat_{factor}", 0)
            sig = "*" if abs(tstat) > 2 else ""
            print(f"    {factor:<6} : {result[key]:>8.3f}  (t={tstat:.2f}) {sig}")


# ---------------------------------------------------------------------------
# 3. ROLLING BETA
# ---------------------------------------------------------------------------
# Beta isn't constant. A stock's sensitivity to the market changes with
# regimes, earnings, sector rotation, etc. Rolling beta shows this drift.
#
# Your risk desk tracks this to make sure PMs aren't drifting from
# their mandate.

def rolling_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """
    Compute rolling beta using a simple rolling covariance / variance.

    beta_t = cov(stock, market)_t / var(market)_t
    """
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ["stock", "market"]

    rolling_cov = aligned["stock"].rolling(window).cov(aligned["market"])
    rolling_var = aligned["market"].rolling(window).var()

    return (rolling_cov / rolling_var).dropna()


def plot_rolling_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    stock_name: str = "",
    window: int = 63,
):
    """Plot rolling beta over time."""
    rb = rolling_beta(stock_returns, market_returns, window)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rb, linewidth=0.8, color="steelblue")
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--", label="Beta = 1")
    ax.axhline(rb.mean(), color="orange", linewidth=0.8, linestyle="--",
               label=f"Mean beta = {rb.mean():.2f}")
    ax.set_title(f"{stock_name} — {window}-day Rolling Beta to SPY")
    ax.set_ylabel("Beta")
    ax.legend()
    plt.tight_layout()
    filename = f"module3_{stock_name}_rolling_beta.png"
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# 4. ALPHA & RESIDUAL ANALYSIS
# ---------------------------------------------------------------------------
# The residuals from a factor regression are the "idiosyncratic" or
# "stock-specific" returns — what's left after you strip out factor
# exposure. If the residuals have a pattern, there's alpha to capture.

def plot_residual_analysis(model_result: dict, name: str = ""):
    """Analyse regression residuals — are they random?"""
    model = model_result["model"]
    residuals = model.resid

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Time series of residuals
    axes[0, 0].plot(residuals, linewidth=0.3, alpha=0.7)
    axes[0, 0].axhline(0, color="red", linewidth=0.5)
    axes[0, 0].set_title("Residuals Over Time")

    # Histogram
    axes[0, 1].hist(residuals, bins=80, density=True, alpha=0.6, color="steelblue")
    axes[0, 1].set_title("Residual Distribution")

    # Autocorrelation — if residuals are autocorrelated, the model is missing something
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=20, ax=axes[1, 0])
    axes[1, 0].set_title("Residual Autocorrelation")

    # Residuals vs fitted values
    axes[1, 1].scatter(model.fittedvalues, residuals, alpha=0.1, s=2)
    axes[1, 1].axhline(0, color="red", linewidth=0.5)
    axes[1, 1].set_xlabel("Fitted Values")
    axes[1, 1].set_ylabel("Residuals")
    axes[1, 1].set_title("Residuals vs Fitted")

    plt.suptitle(f"{name} — Residual Analysis", y=1.02)
    plt.tight_layout()
    filename = f"module3_{name}_residuals.png"
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# 5. INFORMATION RATIO
# ---------------------------------------------------------------------------
# IR = alpha / tracking_error
# Where tracking_error = std(residuals) * sqrt(252)
#
# This tells you: "How much alpha per unit of active risk?"
# It's the key metric for evaluating active managers.
#
# IR > 0.5 is good. IR > 1.0 is exceptional.
# Grinold & Kahn's "Fundamental Law" says: IR = IC * sqrt(breadth)

def information_ratio(model_result: dict) -> float:
    """
    Information ratio from a factor model regression.

    IR = annualised alpha / annualised tracking error
    """
    alpha_annual = model_result["alpha_annual"]
    tracking_error = model_result["model"].resid.std() * np.sqrt(TRADING_DAYS)
    if tracking_error == 0:
        return 0.0
    return alpha_annual / tracking_error


# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  MODULE 3: Factor Models & Regression")
    print("=" * 60)

    start, end = "2018-01-01", "2025-12-31"

    # Fetch data
    spy_returns = fetch_returns("SPY", start, end)
    factors = build_factor_proxies(start, end)

    # Analyse a few stocks
    stocks = ["AAPL", "JPM", "XOM"]
    for ticker in stocks:
        print(f"\n{'='*60}")
        print(f"  Analysing: {ticker}")
        print(f"{'='*60}")

        stock_rets = fetch_returns(ticker, start, end)

        # CAPM
        capm_result = capm_regression(stock_rets, spy_returns)
        print_capm(capm_result, ticker, "SPY")

        # Multi-factor
        mf_result = multifactor_regression(stock_rets, factors)
        print_multifactor(mf_result, ticker)

        # Information ratio
        ir = information_ratio(mf_result)
        print(f"\n    Information Ratio: {ir:.2f}")

        # Rolling beta
        plot_rolling_beta(stock_rets, spy_returns, ticker)

        # Residual analysis
        plot_residual_analysis(mf_result, ticker)

    print(f"\n{'='*60}")
    print("  KEY TAKEAWAYS:")
    print("=" * 60)
    print("  1. Beta measures sensitivity to the market — PMs target specific betas")
    print("  2. Factor models decompose returns into systematic + idiosyncratic")
    print("  3. Alpha is the return NOT explained by factors — it's what hedge funds sell")
    print("  4. R-squared tells you how much of the return is factor-driven")
    print("     (high R² = index-like, low R² = stock-picker)")
    print("  5. The Information Ratio measures alpha efficiency — it's how LPs")
    print("     evaluate managers")
    print()

    print("  EXERCISES:")
    print("  1. Pick a stock your fund owns. What's its CAPM beta? Is it what you expected?")
    print("  2. Add a momentum factor (winners minus losers) to the model. Use")
    print("     MTUM - SPY as a proxy. Does it improve R-squared?")
    print("  3. Compare rolling beta of AAPL vs XOM — which is more stable?")
    print("  4. Find a stock with statistically significant alpha (t > 2).")
    print("     Is it real or is the sample period cherry-picked?")
    print()


if __name__ == "__main__":
    main()
