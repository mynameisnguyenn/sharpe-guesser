"""
Vol-Targeting Strategy Backtest (Paleologo Ch 6)
=================================================

The core idea: instead of holding a fixed portfolio, scale your position size
so that your predicted dollar volatility stays constant. When vol is high,
hold less. When vol is low, hold more.

Paleologo shows this improves Sharpe by 0.1-0.3, reduces max drawdown, and
makes the return distribution more normal (lower kurtosis). The intuition:
high-vol periods tend to have negative expected returns (vol clustering +
leverage effect), so you're effectively avoiding the worst periods.

This is exactly what institutional risk managers do — they set vol targets
and scale positions accordingly. A typical equity vol target is 10-15%.

We compare:
    Strategy A (unmanaged): equal-weight portfolio of 50 stocks, rebalanced monthly
    Strategy B (vol-targeted): same portfolio, but scale by target_vol / forecast_vol
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. EQUAL-WEIGHT PORTFOLIO RETURNS
# ---------------------------------------------------------------------------

def equal_weight_returns(prices: pd.DataFrame) -> pd.Series:
    """
    Compute daily returns of an equal-weight portfolio.

    Each stock gets weight 1/N at every rebalance.
    Daily returns = mean of individual stock returns each day.
    """
    daily_returns = prices.pct_change()
    # Equal-weight = simple average across stocks each day
    portfolio_returns = daily_returns.mean(axis=1)
    portfolio_returns = portfolio_returns.iloc[1:]  # drop first NaN
    portfolio_returns.name = "equal_weight"
    return portfolio_returns


# ---------------------------------------------------------------------------
# 2. VOL-TARGETED RETURNS
# ---------------------------------------------------------------------------

def vol_targeted_returns(
    portfolio_returns: pd.Series,
    vol_forecast: pd.Series,
    target_vol: float = 0.10,
    max_leverage: float = 2.0,
    min_leverage: float = 0.1,
) -> pd.Series:
    """
    Apply vol targeting to a portfolio.

    At each day t:
        leverage_t = target_vol / forecast_vol_t
        managed_return_t = leverage_t * raw_return_t

    The leverage is clipped to [min_leverage, max_leverage] to prevent
    extreme positions. A max_leverage of 2x means you'll never go beyond
    200% invested; min_leverage of 0.1x means you'll always hold at least
    10% of the portfolio.

    Parameters:
        portfolio_returns: daily returns of the unmanaged portfolio
        vol_forecast:      predicted annualized vol (any model's output)
        target_vol:        target annualized vol (default 10%)
        max_leverage:      maximum leverage ratio
        min_leverage:      minimum leverage ratio
    """
    # Align on common dates
    common = portfolio_returns.index.intersection(vol_forecast.index)
    ret = portfolio_returns.loc[common]
    vol = vol_forecast.loc[common]

    # Compute leverage ratio
    leverage = target_vol / vol
    leverage = leverage.clip(lower=min_leverage, upper=max_leverage)

    # Managed returns — shift leverage by 1 to avoid lookahead
    # We use today's forecast to size TOMORROW's position
    leverage_lagged = leverage.shift(1)

    managed = ret * leverage_lagged
    managed = managed.dropna()
    managed.name = "vol_targeted"
    return managed


# ---------------------------------------------------------------------------
# 3. PERFORMANCE METRICS
# ---------------------------------------------------------------------------

def compute_performance(
    returns: pd.Series,
    rf: float = 0.05,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute standard performance metrics for a daily return series.

    Returns dict with: annual_return, annual_vol, sharpe, sortino,
    max_drawdown, skewness, kurtosis.
    """
    clean = returns.dropna()
    if len(clean) < 10:
        return {}

    ann_ret = clean.mean() * periods_per_year
    ann_vol = clean.std() * np.sqrt(periods_per_year)

    excess = ann_ret - rf
    sharpe = excess / ann_vol if ann_vol > 1e-10 else 0.0

    # Sortino — downside deviation
    downside = clean[clean < 0]
    dd = downside.std() * np.sqrt(periods_per_year)
    sortino = excess / dd if dd > 1e-10 and not np.isnan(dd) else 0.0

    # Max drawdown
    cumulative = (1 + clean).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "skewness": clean.skew(),
        "kurtosis": clean.kurtosis(),
    }


# ---------------------------------------------------------------------------
# 4. RUN VOL-TARGETING BACKTEST
# ---------------------------------------------------------------------------

def run_vol_target_backtest(
    prices: pd.DataFrame,
    vol_forecast: pd.Series,
    target_vol: float = 0.10,
) -> dict:
    """
    Full vol-targeting backtest comparing managed vs unmanaged portfolios.

    Parameters:
        prices:       DataFrame of daily adj close for universe stocks
        vol_forecast: Series of annualized vol forecasts (e.g., EWMA on portfolio)
        target_vol:   target annualized vol (default 10%)

    Returns:
        dict with 'unmanaged' and 'managed' sub-dicts, each containing:
        - returns: daily return Series
        - cumulative: cumulative return Series
        - rolling_vol: 63-day rolling vol
        - metrics: performance dict
    """
    # Unmanaged portfolio
    raw_returns = equal_weight_returns(prices)

    # Vol-targeted portfolio
    managed_returns = vol_targeted_returns(
        raw_returns, vol_forecast,
        target_vol=target_vol,
    )

    # Align to common dates for fair comparison
    common = raw_returns.index.intersection(managed_returns.index)
    raw_aligned = raw_returns.loc[common]
    managed_aligned = managed_returns.loc[common]

    results = {}
    for label, ret in [("unmanaged", raw_aligned), ("managed", managed_aligned)]:
        cumulative = (1 + ret).cumprod()
        rolling_vol = ret.rolling(63).std() * np.sqrt(252)
        metrics = compute_performance(ret)

        results[label] = {
            "returns": ret,
            "cumulative": cumulative,
            "rolling_vol": rolling_vol,
            "metrics": metrics,
        }

    # Print comparison
    print("\n" + "=" * 70)
    print(f"  Vol-Targeting Backtest (target = {target_vol:.0%})")
    print("=" * 70)

    for label in ["unmanaged", "managed"]:
        m = results[label]["metrics"]
        print(f"\n  {label.title()}:")
        print(f"    Annual Return:  {m['annual_return']:.2%}")
        print(f"    Annual Vol:     {m['annual_vol']:.2%}")
        print(f"    Sharpe:         {m['sharpe']:.3f}")
        print(f"    Sortino:        {m['sortino']:.3f}")
        print(f"    Max Drawdown:   {m['max_drawdown']:.2%}")
        print(f"    Skewness:       {m['skewness']:.3f}")
        print(f"    Kurtosis:       {m['kurtosis']:.3f}")

    # Improvement summary
    um = results["unmanaged"]["metrics"]
    mm = results["managed"]["metrics"]
    print(f"\n  Sharpe improvement:     {mm['sharpe'] - um['sharpe']:+.3f}")
    print(f"  Max DD improvement:     {mm['max_drawdown'] - um['max_drawdown']:+.2%}")
    print(f"  Kurtosis reduction:     {mm['kurtosis'] - um['kurtosis']:+.3f}")
    print("=" * 70 + "\n")

    return results
