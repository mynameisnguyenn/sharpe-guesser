"""
Feature Engineering: Firm-Level Characteristics
================================================

This module constructs the predictor variables (features) used to forecast
stock returns. In the Kelly paper, Gu, Kelly, and Xiu (2020) use 94 firm-level
characteristics plus their interactions — totaling ~900 features. We use a
simplified set of ~15 features that capture the most important signal groups:

    1. Momentum (1m, 6m, 12m) — the strongest predictor in the paper
    2. Short-term reversal — contrarian signal at the 1-month horizon
    3. Realized volatility — idiosyncratic risk
    4. Log market cap — size effect (small stocks earn higher returns)
    5. Volume / turnover — liquidity signal

Each feature function operates on a panel (DataFrame with DatetimeIndex and
ticker columns) and returns a panel of the same shape. NaN values arise
naturally from lookback windows — downstream code handles them.

Why these features? The paper's variable importance analysis shows that
momentum, volatility, and liquidity dominate the other 90+ characteristics.
We focus on what matters most.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. MOMENTUM SIGNALS
# ---------------------------------------------------------------------------
# Momentum is the single most important predictor in Gu, Kelly, Xiu (2020).
# Stocks that went up tend to keep going up (at the 6-12 month horizon).
# At the 1-month horizon, the effect reverses — this is short-term reversal.
#
# The "skip the most recent month" convention for 6m and 12m momentum is
# standard in the academic literature. The most recent month captures
# short-term reversal (a different signal), so we exclude it to keep
# momentum and reversal as separate, cleaner signals.

def momentum_1m(prices: pd.DataFrame) -> pd.DataFrame:
    """
    1-month trailing return (short-term reversal signal).

    This captures the short-term reversal effect: stocks that went up last
    month tend to mean-revert. In Gu, Kelly, Xiu (2020), short-term reversal
    is the second most important predictor after momentum.
    """
    # 21 trading days ~ 1 month
    return prices.pct_change(periods=21)


def momentum_6m(prices: pd.DataFrame) -> pd.DataFrame:
    """
    6-month trailing return, skipping the most recent month.

    Momentum is the strongest predictor in Gu, Kelly, Xiu (2020). We skip
    the most recent month to avoid contamination from the short-term reversal
    effect, which operates in the opposite direction.
    """
    # 126 trading days ~ 6 months, shift by 21 to skip most recent month
    total_return = prices.pct_change(periods=126)
    recent_return = prices.pct_change(periods=21)
    # (1 + r_6m) / (1 + r_1m) - 1 gives the return from t-126 to t-21
    return (1 + total_return) / (1 + recent_return) - 1


def momentum_12m(prices: pd.DataFrame) -> pd.DataFrame:
    """
    12-month trailing return, skipping the most recent month.

    The classic Jegadeesh and Titman (1993) momentum signal. The Kelly paper
    confirms its importance even after controlling for dozens of other
    characteristics via machine learning.
    """
    # 252 trading days ~ 12 months, shift by 21 to skip most recent month
    total_return = prices.pct_change(periods=252)
    recent_return = prices.pct_change(periods=21)
    return (1 + total_return) / (1 + recent_return) - 1


# ---------------------------------------------------------------------------
# 2. VOLATILITY
# ---------------------------------------------------------------------------
# Realized volatility is a proxy for idiosyncratic risk. The low-volatility
# anomaly says that low-vol stocks earn higher risk-adjusted returns than
# high-vol stocks — which contradicts basic CAPM. The Kelly paper uses
# volatility as one of its important features.
#
# On your risk desk, you compute realized vol every day. Same idea here,
# just applied to each stock individually.

def realized_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Trailing realized volatility (annualized).

    Rolling standard deviation of daily returns over the lookback window,
    annualized by multiplying by sqrt(252). This is the same calculation
    your risk system does for single-stock vol.
    """
    return returns.rolling(window=window).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# 3. SIZE
# ---------------------------------------------------------------------------
# The size effect (Fama and French, 1993): small stocks earn higher returns
# than large stocks, on average. We proxy market cap with price x volume
# (since shares outstanding requires a separate data source). Log transform
# compresses the distribution — market caps span several orders of magnitude.

def log_market_cap(prices: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """
    Log market capitalization: log(price x shares outstanding).

    If shares outstanding data is unavailable, pass a proxy like average daily
    dollar volume. The key is to capture the cross-sectional size ranking,
    not the exact dollar value. Log transform is standard because market caps
    range from ~$1B to ~$3T in the S&P 500.
    """
    market_cap = prices * shares
    # Avoid log(0) or log(negative)
    market_cap = market_cap.clip(lower=1.0)
    return np.log(market_cap)


def log_dollar_volume(prices: pd.DataFrame, volumes: pd.DataFrame,
                      window: int = 21) -> pd.DataFrame:
    """
    Log average daily dollar volume — proxy for size when shares outstanding
    data is unavailable. Dollar volume = price x volume.
    """
    dollar_volume = prices * volumes
    avg_dollar_volume = dollar_volume.rolling(window=window).mean()
    avg_dollar_volume = avg_dollar_volume.clip(lower=1.0)
    return np.log(avg_dollar_volume)


# ---------------------------------------------------------------------------
# 4. LIQUIDITY / TURNOVER
# ---------------------------------------------------------------------------
# Turnover (volume / shares outstanding) captures liquidity and investor
# attention. The Kelly paper finds that liquidity-related signals are among
# the top predictors. High turnover often signals disagreement or news.

def turnover(volumes: pd.DataFrame, shares: pd.DataFrame,
             window: int = 21) -> pd.DataFrame:
    """
    Average daily turnover: mean(volume / shares_outstanding) over a window.

    Higher turnover = more liquid, more attention. In the Kelly paper,
    turnover interacts with momentum to create powerful combined signals.
    """
    daily_turnover = volumes / shares
    return daily_turnover.rolling(window=window).mean()


def volume_trend(volumes: pd.DataFrame, short: int = 5,
                 long: int = 21) -> pd.DataFrame:
    """
    Volume trend: ratio of short-term to long-term average volume.

    Values > 1 mean volume is increasing (potential catalyst or regime change).
    This is a simple version of the volume signals in the Kelly paper.
    """
    short_avg = volumes.rolling(window=short).mean()
    long_avg = volumes.rolling(window=long).mean()
    # Avoid division by zero
    long_avg = long_avg.replace(0, np.nan)
    return short_avg / long_avg


# ---------------------------------------------------------------------------
# 5. ADDITIONAL SIGNALS
# ---------------------------------------------------------------------------

def high_low_range(highs: pd.DataFrame, lows: pd.DataFrame,
                   window: int = 21) -> pd.DataFrame:
    """
    Average daily high-low range as a fraction of the closing price.

    This is a measure of intraday volatility. The Kelly paper includes
    several price-range-based features; this is a simplified version.
    """
    daily_range = (highs - lows) / lows
    return daily_range.rolling(window=window).mean()


def max_return(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Maximum daily return over the trailing window.

    Stocks with extreme recent positive returns tend to underperform
    (lottery effect). This is one of the newer signals in the Kelly paper.
    """
    return returns.rolling(window=window).max()


# ---------------------------------------------------------------------------
# 6. BUILD ALL FEATURES — ORCHESTRATE
# ---------------------------------------------------------------------------

def build_features(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construct all features from price and volume data.

    Parameters:
        prices:   Daily adjusted close prices (DatetimeIndex, tickers as columns)
        volumes:  Daily trading volumes (same shape as prices)
        returns:  Daily returns (optional — computed from prices if not provided)

    Returns:
        A MultiIndex DataFrame: (date, ticker) as index, features as columns.
        This "long" format is what the model expects: one row per stock-date.
    """
    if returns is None:
        returns = prices.pct_change()

    # Compute each feature as a wide DataFrame (dates x tickers)
    features = {
        "mom_1m": momentum_1m(prices),
        "mom_6m": momentum_6m(prices),
        "mom_12m": momentum_12m(prices),
        "realized_vol": realized_volatility(returns, window=21),
        "log_dollar_vol": log_dollar_volume(prices, volumes, window=21),
        "volume_trend": volume_trend(volumes, short=5, long=21),
        "max_return": max_return(returns, window=21),
    }

    # Stack each feature into long format and combine
    feature_frames = []
    for name, df in features.items():
        stacked = df.stack()
        stacked.name = name
        feature_frames.append(stacked)

    result = pd.concat(feature_frames, axis=1)
    result.index.names = ["date", "ticker"]

    # Drop rows where ALL features are NaN (from different lookback windows)
    result = result.dropna(how="all")

    return result


# ---------------------------------------------------------------------------
# 7. FEATURE INTERACTIONS
# ---------------------------------------------------------------------------
# The Kelly paper's key innovation: ML models capture interactions between
# features that linear models miss. For example, "momentum works for large
# caps but reverses for micro caps" is a momentum x size interaction.
# The paper uses all pairwise interactions of 94 features (~4,400 terms).
# We do the same with our 7 base features = 21 interaction terms.

def build_interactions(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pairwise interaction terms (products) between all features.

    This is a simple way to give linear models access to the nonlinear
    signal that tree models capture automatically. The Kelly paper finds
    that interactions substantially improve elastic net performance.
    """
    cols = features_df.columns.tolist()
    interactions = {}
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            interactions[f"{c1}_x_{c2}"] = features_df[c1] * features_df[c2]

    interaction_df = pd.DataFrame(interactions, index=features_df.index)
    return pd.concat([features_df, interaction_df], axis=1)
