"""
Realized Volatility: computation and feature engineering
========================================================

Core functions for computing realized vol, forward vol targets,
HAR-RV features (Corsi 2009), and vol cones.

All inputs are daily return Series with DatetimeIndex.
Annualization factor: sqrt(252).
"""

import numpy as np
import pandas as pd


ANN_FACTOR = np.sqrt(252)


def realized_vol(returns: pd.Series, window: int = 22) -> pd.Series:
    """Annualized realized volatility — rolling std of returns * sqrt(252)."""
    return returns.rolling(window).std() * ANN_FACTOR


def forward_rv(returns: pd.Series, window: int = 22) -> pd.Series:
    """
    Forward realized vol — the prediction target.

    forward_rv[t] = realized vol of returns[t+1 : t+window].
    This is the vol that WILL BE realized over the next `window` days.
    Uses shift(-window) so the target is aligned to the prediction date.
    """
    return realized_vol(returns, window).shift(-window)


def har_features(returns: pd.Series) -> pd.DataFrame:
    """
    HAR-RV features (Corsi 2009).

    The Heterogeneous Autoregressive model decomposes vol into three
    frequencies — daily, weekly, monthly — to capture different trader
    horizons.

    Returns DataFrame with columns:
        rv_d: daily RV — sqrt(r_t^2 * 252), captures single-day shocks
        rv_w: weekly RV — 5-day rolling std * sqrt(252)
        rv_m: monthly RV — 22-day rolling std * sqrt(252)
    """
    rv_d = np.sqrt(returns ** 2 * 252)
    rv_w = returns.rolling(5).std() * ANN_FACTOR
    rv_m = returns.rolling(22).std() * ANN_FACTOR

    return pd.DataFrame({
        "rv_d": rv_d,
        "rv_w": rv_w,
        "rv_m": rv_m,
    }, index=returns.index)


def vol_cone(
    returns: pd.Series,
    windows: list[int] = None,
) -> pd.DataFrame:
    """
    Vol cone — percentiles of realized vol at different horizons.

    For each window, computes realized vol over the full history, then
    reports the 10th/25th/50th/75th/90th percentiles. Shows where current
    vol sits relative to its historical distribution.

    Returns DataFrame: index = windows, columns = percentile labels.
    """
    if windows is None:
        windows = [5, 10, 22, 63, 126, 252]

    percentiles = [10, 25, 50, 75, 90]
    rows = []

    for w in windows:
        rv = realized_vol(returns, window=w).dropna()
        pcts = np.percentile(rv, percentiles)
        rows.append(pcts)

    return pd.DataFrame(
        rows,
        index=windows,
        columns=[f"{p}th" for p in percentiles],
    )
