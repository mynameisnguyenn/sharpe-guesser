"""
Portfolio Construction: Decile Sorts and Long-Short Portfolios
================================================================

This module turns model predictions into tradeable portfolios. The logic is
straightforward: rank stocks by predicted return, go long the top decile,
short the bottom decile. This "long-short spread" isolates the model's
ability to predict the cross-section of returns, removing market exposure.

This is exactly how the Kelly paper evaluates its models — not by R-squared
(which is tiny for monthly stock returns), but by whether the predictions
produce economically meaningful portfolio returns.

The performance metrics here are the same ones from modules/module_2_risk_metrics.py
in the parent project. If you've already worked through that module, these
calculations will be familiar.
"""

import numpy as np
import pandas as pd


TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 1. RANK STOCKS INTO QUANTILES
# ---------------------------------------------------------------------------
# At each date, we rank stocks by predicted return and assign them to deciles.
# Decile 10 = highest predicted return (long), decile 1 = lowest (short).
# This is the standard "portfolio sort" methodology from the asset pricing
# literature. On your risk desk, you'd call this a "factor quintile analysis."

def rank_stocks(
    predictions: pd.Series,
    n_quantiles: int = 10,
) -> pd.Series:
    """
    Rank stocks into quantiles by predicted return at each date.

    Parameters:
        predictions: Series with MultiIndex (date, ticker), values = predicted returns
        n_quantiles: Number of buckets (default 10 = deciles)

    Returns:
        Series of quantile labels (1 = lowest predicted, n = highest predicted)
    """
    def _rank_date(group: pd.Series) -> pd.Series:
        """Rank within a single date."""
        try:
            return pd.qcut(group, q=n_quantiles, labels=False, duplicates="drop") + 1
        except ValueError:
            # Not enough unique values for the requested number of quantiles
            return pd.Series(np.nan, index=group.index)

    # Group by date (first level of MultiIndex) and rank within each date
    rankings = predictions.groupby(level=0).apply(_rank_date)

    # Fix any MultiIndex issues from groupby
    if isinstance(rankings.index, pd.MultiIndex) and rankings.index.nlevels > 2:
        rankings = rankings.droplevel(0)

    return rankings


# ---------------------------------------------------------------------------
# 2. LONG-SHORT PORTFOLIO RETURNS
# ---------------------------------------------------------------------------
# The long-short portfolio goes long the top decile and short the bottom
# decile. This is a dollar-neutral portfolio — you're not exposed to the
# market direction, only to your model's ability to rank stocks.
#
# Think of it this way: if your model is good, it puts winners in decile 10
# and losers in decile 1. The spread between them is your alpha.

def long_short_returns(
    returns: pd.Series,
    rankings: pd.Series,
    long_quantile: int | None = None,
    short_quantile: int = 1,
) -> pd.Series:
    """
    Compute returns of a long-short portfolio (long top decile, short bottom).

    Parameters:
        returns:        Series with MultiIndex (date, ticker), values = actual returns
        rankings:       Series with same index, values = quantile labels (1..n)
        long_quantile:  Which quantile to go long (default: max = top decile)
        short_quantile: Which quantile to go short (default: 1 = bottom decile)

    Returns:
        Series indexed by date with the long-short spread return each period.
    """
    if long_quantile is None:
        long_quantile = int(rankings.max())

    # Equal-weight within each leg
    long_mask = rankings == long_quantile
    short_mask = rankings == short_quantile

    long_returns = returns[long_mask].groupby(level=0).mean()
    short_returns = returns[short_mask].groupby(level=0).mean()

    # Long-short = long leg minus short leg
    spread = long_returns - short_returns

    return spread.dropna()


# ---------------------------------------------------------------------------
# 3. PERFORMANCE METRICS
# ---------------------------------------------------------------------------
# These metrics are the same ones from modules/module_2_risk_metrics.py in
# the parent project. If you've worked through Module 2, you already know
# how to compute these. The Kelly paper reports annualized Sharpe ratios
# for its long-short portfolios — that's the primary evaluation metric.

def compute_performance(
    strategy_returns: pd.Series,
    rf: float = 0.05,
    periods_per_year: int = 12,
) -> dict[str, float]:
    """
    Compute standard performance metrics for a return series.

    Parameters:
        strategy_returns: Series of periodic returns (monthly or daily)
        rf:               Annual risk-free rate (default 5%)
        periods_per_year: 12 for monthly, 252 for daily

    Returns:
        Dict with: annual_return, annual_vol, sharpe, sortino, max_drawdown
    """
    rf_per_period = rf / periods_per_year
    excess = strategy_returns - rf_per_period

    # Annualize
    annual_return = strategy_returns.mean() * periods_per_year
    annual_vol = strategy_returns.std() * np.sqrt(periods_per_year)

    # Sharpe ratio
    sharpe = (annual_return - rf) / annual_vol if annual_vol > 1e-10 else 0.0

    # Sortino ratio — only penalize downside volatility
    downside = excess[excess < 0]
    downside_std = downside.std() * np.sqrt(periods_per_year)
    if downside_std > 1e-10 and not np.isnan(downside_std):
        sortino = (annual_return - rf) / downside_std
    else:
        sortino = 0.0

    # Max drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
    }


# ---------------------------------------------------------------------------
# 4. QUANTILE RETURN ANALYSIS
# ---------------------------------------------------------------------------
# Beyond just long-short, it's useful to see returns across ALL deciles.
# A good model produces a monotonic pattern: decile 1 has the lowest return,
# decile 10 the highest, with a smooth gradient in between.

def quantile_returns(
    returns: pd.Series,
    rankings: pd.Series,
) -> pd.DataFrame:
    """
    Compute average returns for each quantile — the "portfolio sort" table.

    A good model shows a monotonic increase from quantile 1 to quantile N.
    This is the standard way to evaluate cross-sectional return predictions
    in the academic literature.
    """
    combined = pd.DataFrame({"return": returns, "quantile": rankings}).dropna()

    summary = combined.groupby("quantile")["return"].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["mean_return", "std_return", "n_stocks"]

    # Annualize (assuming monthly)
    summary["annual_return"] = summary["mean_return"] * 12
    summary["annual_vol"] = summary["std_return"] * np.sqrt(12)

    return summary
