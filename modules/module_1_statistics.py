"""
Module 1: Statistics for Quant Finance
=======================================

Before you can build models, you need to understand the statistical
properties of financial returns. This module covers the concepts that
show up every day on a risk desk — but now you'll compute them yourself.

Topics:
    1. Descriptive statistics of returns
    2. Distribution shapes: skewness & kurtosis (fat tails!)
    3. Normal vs. real return distributions
    4. Correlation & covariance matrices
    5. Stationarity — why it matters for everything

Run:
    python -m modules.module_1_statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf


# ---------------------------------------------------------------------------
# 1. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------
# You've seen these on every risk report. Now build them.

def return_statistics(daily_returns: pd.Series) -> dict:
    """
    Compute the descriptive statistics a risk desk cares about.

    These are the numbers you see on a one-pager for any fund or strategy.
    """
    return {
        "mean_daily": daily_returns.mean(),
        "mean_annual": daily_returns.mean() * 252,
        "vol_daily": daily_returns.std(),
        "vol_annual": daily_returns.std() * np.sqrt(252),
        "skewness": daily_returns.skew(),
        "kurtosis": daily_returns.kurtosis(),       # excess kurtosis
        "min": daily_returns.min(),
        "max": daily_returns.max(),
        "median": daily_returns.median(),
        "pct_positive": (daily_returns > 0).mean(),  # win rate
    }


def print_statistics(stats_dict: dict, name: str = ""):
    """Pretty-print return statistics."""
    print(f"\n{'='*50}")
    print(f"  Return Statistics: {name}")
    print(f"{'='*50}")
    print(f"  Mean daily return  : {stats_dict['mean_daily']:>10.4%}")
    print(f"  Mean annual return : {stats_dict['mean_annual']:>10.2%}")
    print(f"  Daily volatility   : {stats_dict['vol_daily']:>10.4%}")
    print(f"  Annual volatility  : {stats_dict['vol_annual']:>10.2%}")
    print(f"  Skewness           : {stats_dict['skewness']:>10.2f}")
    print(f"  Excess kurtosis    : {stats_dict['kurtosis']:>10.2f}")
    print(f"  Worst day          : {stats_dict['min']:>10.2%}")
    print(f"  Best day           : {stats_dict['max']:>10.2%}")
    print(f"  Median daily       : {stats_dict['median']:>10.4%}")
    print(f"  % positive days    : {stats_dict['pct_positive']:>10.1%}")


# ---------------------------------------------------------------------------
# 2. SKEWNESS & KURTOSIS — WHY "NORMAL" IS A LIE
# ---------------------------------------------------------------------------
# Finance textbooks assume returns are normally distributed.
# They're not. Real returns have:
#   - Negative skew (big down days are bigger than big up days)
#   - Fat tails / excess kurtosis (extreme events happen way more than
#     a normal distribution predicts)
#
# This is why your fund's risk models use stressed scenarios and not
# just standard deviation.

def plot_distribution_vs_normal(daily_returns: pd.Series, name: str = ""):
    """
    Overlay actual return distribution against a fitted normal.

    The gap between these two — especially in the tails — is where
    real risk lives. This is why VaR models blow up.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram vs normal
    mu, sigma = daily_returns.mean(), daily_returns.std()
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    normal_pdf = stats.norm.pdf(x, mu, sigma)

    axes[0].hist(daily_returns, bins=100, density=True, alpha=0.6,
                 color="steelblue", label="Actual returns")
    axes[0].plot(x, normal_pdf, "r-", linewidth=2, label="Normal fit")
    axes[0].set_title(f"{name} — Return Distribution vs Normal")
    axes[0].set_xlabel("Daily Return")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # QQ plot — if returns were normal, dots would sit on the red line.
    # Deviations in the tails = fat tails.
    stats.probplot(daily_returns.dropna(), dist="norm", plot=axes[1])
    axes[1].set_title(f"{name} — QQ Plot (Normal)")
    axes[1].get_lines()[0].set_markerfacecolor("steelblue")
    axes[1].get_lines()[0].set_markersize(3)

    plt.tight_layout()
    filename = f"module1_{name.replace(' ', '_')}_distribution.png"
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# 3. JARQUE-BERA TEST — IS IT NORMAL?
# ---------------------------------------------------------------------------
# This is the formal hypothesis test. If p-value < 0.05, the returns
# are NOT normally distributed. Spoiler: they almost never are.

def test_normality(daily_returns: pd.Series, name: str = "") -> dict:
    """Run normality tests on a return series."""
    jb_stat, jb_p = stats.jarque_bera(daily_returns.dropna())
    sw_stat, sw_p = stats.shapiro(daily_returns.dropna()[:5000])  # shapiro has size limit

    result = {
        "jarque_bera_stat": jb_stat,
        "jarque_bera_p": jb_p,
        "shapiro_wilk_stat": sw_stat,
        "shapiro_wilk_p": sw_p,
        "is_normal_jb": jb_p > 0.05,
        "is_normal_sw": sw_p > 0.05,
    }

    print(f"\n  Normality tests for {name}:")
    print(f"    Jarque-Bera  : stat={jb_stat:.1f}, p={jb_p:.6f}  "
          f"{'Normal' if result['is_normal_jb'] else 'NOT Normal'}")
    print(f"    Shapiro-Wilk : stat={sw_stat:.4f}, p={sw_p:.6f}  "
          f"{'Normal' if result['is_normal_sw'] else 'NOT Normal'}")

    return result


# ---------------------------------------------------------------------------
# 4. CORRELATION & COVARIANCE
# ---------------------------------------------------------------------------
# At a multi-PM fund, understanding correlation between books is critical.
# Low correlation = diversification = the PM can run more gross.

def correlation_analysis(tickers: list, start: str, end: str):
    """
    Download multiple assets and compute their correlation matrix.

    This is the foundation of portfolio construction: which assets
    move together and which provide diversification.
    """
    df = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    returns = df.pct_change().dropna()

    corr = returns.corr()
    cov = returns.cov() * 252  # annualised covariance

    print("\n  Correlation Matrix (daily returns):")
    print(corr.round(2).to_string(index=True))

    print("\n  Annualised Covariance Matrix:")
    print(cov.round(4).to_string(index=True))

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45)
    ax.set_yticklabels(tickers)

    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")

    plt.colorbar(im)
    ax.set_title("Return Correlation Matrix")
    plt.tight_layout()
    filename = "module1_correlation_heatmap.png"
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"\n  Saved: {filename}")

    return corr, cov


# ---------------------------------------------------------------------------
# 5. STATIONARITY — THE MOST IMPORTANT CONCEPT YOU'VE NEVER HEARD OF
# ---------------------------------------------------------------------------
# Prices are NOT stationary (they trend up/down). Returns generally ARE.
# If you try to model non-stationary data as stationary, your model
# will produce garbage. This is why quants always work with returns,
# not prices.
#
# The Augmented Dickey-Fuller (ADF) test checks for stationarity.
# p < 0.05 means the series IS stationary.

def test_stationarity(series: pd.Series, name: str = "") -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    You want: returns to be stationary (p < 0.05), prices to not be.
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series.dropna(), autolag="AIC")

    output = {
        "adf_statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "is_stationary": result[1] < 0.05,
    }

    print(f"\n  Stationarity test (ADF) for {name}:")
    print(f"    ADF statistic : {result[0]:.4f}")
    print(f"    p-value       : {result[1]:.6f}")
    print(f"    Lags used     : {result[2]}")
    print(f"    Result        : {'Stationary' if output['is_stationary'] else 'NOT Stationary'}")

    return output


# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  MODULE 1: Statistics for Quant Finance")
    print("=" * 60)

    # Fetch SPY data
    ticker = "SPY"
    prices = yf.download(ticker, start="2018-01-01", end="2025-12-31", progress=False)["Close"].squeeze()
    returns = prices.pct_change().dropna()

    # 1. Descriptive stats
    s = return_statistics(returns)
    print_statistics(s, ticker)

    # 2. Distribution analysis
    print("\n--- Distribution Analysis ---")
    plot_distribution_vs_normal(returns, ticker)

    # 3. Normality tests
    test_normality(returns, ticker)

    # 4. Correlation across assets
    print("\n--- Cross-Asset Correlation ---")
    tickers = ["SPY", "TLT", "GLD", "QQQ", "XLE"]
    correlation_analysis(tickers, "2018-01-01", "2025-12-31")

    # 5. Stationarity
    print("\n--- Stationarity ---")
    test_stationarity(prices, f"{ticker} prices")
    test_stationarity(returns, f"{ticker} returns")

    print("\n" + "=" * 60)
    print("  KEY TAKEAWAYS:")
    print("=" * 60)
    print("  1. Real returns have fat tails — normal assumptions understate risk")
    print("  2. Negative skew means crashes are sharper than rallies")
    print("  3. Correlation is not constant — it spikes in crises (when you")
    print("     need diversification most)")
    print("  4. Prices are non-stationary; returns are stationary — always")
    print("     model returns, not prices")
    print()

    print("  EXERCISES:")
    print("  1. Run this on your fund's benchmark — do the stats match your intuition?")
    print("  2. Compare kurtosis of SPY vs a single stock — which has fatter tails?")
    print("  3. Compute rolling 60-day correlation between SPY and TLT — watch it")
    print("     spike negative in March 2020")
    print()


if __name__ == "__main__":
    main()
