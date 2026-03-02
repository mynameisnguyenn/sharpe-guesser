"""
Sample Data Generator — Offline Fallback for Learning
======================================================

Generates realistic synthetic market data so the learning modules work
even without internet access (no Yahoo Finance needed).

The data mimics real asset behaviour:
    - SPY:  moderate return, moderate vol (~15%)
    - QQQ:  higher return, higher vol (~20%)
    - AAPL: high return, high vol (~28%)
    - MSFT: high return, moderate vol (~24%)
    - GOOGL: moderate return, high vol (~26%)
    - AMZN: high return, very high vol (~30%)
    - JPM:  moderate return, moderate vol (~22%)
    - GS:   moderate return, high vol (~25%)
    - BAC:  low return, high vol (~26%)
    - XOM:  low return, high vol (~28%)
    - CVX:  low return, high vol (~27%)
    - TLT:  low return, low vol (~12%), negative correlation to equities
    - IEF:  low return, very low vol (~7%)
    - SHY:  near-zero return, very low vol (~1.5%)
    - GLD:  low-moderate return, moderate vol (~14%)
    - SLV:  low return, high vol (~25%)
    - IWM:  moderate return, higher vol (~20%)
    - DIA:  moderate return, moderate vol (~15%)
    - IWD:  moderate return, moderate vol (~17%) — value ETF
    - IWF:  higher return, higher vol (~19%) — growth ETF
    - XLF:  moderate return, moderate vol (~20%)
    - XLE:  low return, high vol (~28%)
    - XLK:  high return, moderate vol (~20%)
    - XLV:  moderate return, low vol (~15%)
    - JNJ:  low-moderate return, low vol (~16%)
    - PG:   low return, low vol (~14%)
    - KO:   low return, low vol (~14%)
    - PEP:  low return, low vol (~14%)
    - MTUM: moderate return, moderate vol (~17%) — momentum ETF

Usage:
    from sample_data import get_prices, get_multi_prices

    prices = get_prices("SPY", "2020-01-01", "2025-12-31")
    multi  = get_multi_prices(["SPY", "TLT", "GLD"], "2020-01-01", "2025-12-31")
"""

import numpy as np
import pandas as pd

# Seed for reproducibility — same data every run so results are consistent
_SEED = 42

# Asset parameters: (annual_return, annual_vol, correlation_group)
# Correlation groups: 0=equity_core, 1=equity_tech, 2=equity_financial,
#                     3=equity_energy, 4=bonds, 5=commodities, 6=equity_defensive
_ASSET_PARAMS = {
    "SPY":  (0.12, 0.15, 0),
    "QQQ":  (0.16, 0.20, 1),
    "IWM":  (0.09, 0.20, 0),
    "DIA":  (0.10, 0.15, 0),
    "AAPL": (0.22, 0.28, 1),
    "MSFT": (0.20, 0.24, 1),
    "GOOGL":(0.15, 0.26, 1),
    "AMZN": (0.18, 0.30, 1),
    "JPM":  (0.11, 0.22, 2),
    "GS":   (0.10, 0.25, 2),
    "BAC":  (0.07, 0.26, 2),
    "XOM":  (0.06, 0.28, 3),
    "CVX":  (0.07, 0.27, 3),
    "TLT":  (0.02, 0.12, 4),
    "IEF":  (0.02, 0.07, 4),
    "SHY":  (0.015, 0.015, 4),
    "GLD":  (0.07, 0.14, 5),
    "SLV":  (0.04, 0.25, 5),
    "IWD":  (0.08, 0.17, 0),
    "IWF":  (0.15, 0.19, 1),
    "XLF":  (0.09, 0.20, 2),
    "XLE":  (0.05, 0.28, 3),
    "XLK":  (0.18, 0.20, 1),
    "XLV":  (0.10, 0.15, 6),
    "JNJ":  (0.06, 0.16, 6),
    "PG":   (0.07, 0.14, 6),
    "KO":   (0.06, 0.14, 6),
    "PEP":  (0.07, 0.14, 6),
    "MTUM": (0.11, 0.17, 0),
}

# Correlation between groups (symmetric)
# Groups: 0=eq_core, 1=eq_tech, 2=eq_fin, 3=eq_energy, 4=bonds, 5=commod, 6=eq_defensive
_GROUP_CORR = np.array([
    [1.00, 0.85, 0.80, 0.60, -0.20, 0.10, 0.65],  # 0: equity_core
    [0.85, 1.00, 0.70, 0.45, -0.15, 0.05, 0.55],  # 1: equity_tech
    [0.80, 0.70, 1.00, 0.55, -0.10, 0.10, 0.60],  # 2: equity_financial
    [0.60, 0.45, 0.55, 1.00, -0.05, 0.35, 0.40],  # 3: equity_energy
    [-0.20, -0.15, -0.10, -0.05, 1.00, 0.15, 0.00],  # 4: bonds
    [0.10, 0.05, 0.10, 0.35, 0.15, 1.00, 0.10],  # 5: commodities
    [0.65, 0.55, 0.60, 0.40, 0.00, 0.10, 1.00],  # 6: equity_defensive
])


def _generate_all_prices(start: str = "2014-01-01", end: str = "2025-12-31") -> dict:
    """
    Generate correlated daily price series for all assets.

    Uses a Cholesky decomposition to create correlated returns,
    then builds price series from a starting price of $100.
    Includes a simulated COVID crash (March 2020) and recovery.
    """
    rng = np.random.RandomState(_SEED)

    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    tickers = list(_ASSET_PARAMS.keys())
    n_assets = len(tickers)

    # Build correlation matrix for individual assets based on group membership
    corr = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            gi = _ASSET_PARAMS[tickers[i]][2]
            gj = _ASSET_PARAMS[tickers[j]][2]
            base_corr = _GROUP_CORR[gi, gj]
            # Add some noise to make within-group correlations not identical
            noise = rng.uniform(-0.05, 0.05)
            corr[i, j] = np.clip(base_corr + noise, -0.95, 0.95)
            corr[j, i] = corr[i, j]

    # Make it positive semi-definite via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0.001)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Renormalize to correlation matrix
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

    # Cholesky decomposition for correlated normals
    L = np.linalg.cholesky(corr)

    # Generate correlated daily returns
    z = rng.randn(n_days, n_assets)
    correlated_z = z @ L.T

    # Scale to match each asset's parameters
    daily_returns = np.zeros((n_days, n_assets))
    for i, ticker in enumerate(tickers):
        ann_ret, ann_vol, _ = _ASSET_PARAMS[ticker]
        daily_mu = ann_ret / 252
        daily_sigma = ann_vol / np.sqrt(252)
        daily_returns[:, i] = daily_mu + daily_sigma * correlated_z[:, i]

    # Add a COVID-like crash around March 2020
    crash_start = pd.Timestamp("2020-02-20")
    crash_bottom = pd.Timestamp("2020-03-23")
    recovery_end = pd.Timestamp("2020-08-01")

    for i, ticker in enumerate(tickers):
        _, ann_vol, group = _ASSET_PARAMS[ticker]

        for d_idx, date in enumerate(dates):
            if crash_start <= date <= crash_bottom:
                # Crash phase — equities drop, bonds rally
                if group in [0, 1, 2, 3, 6]:  # equities
                    crash_intensity = ann_vol * 0.15  # higher vol = bigger crash
                    daily_returns[d_idx, i] -= crash_intensity * rng.uniform(0.5, 1.5)
                elif group == 4:  # bonds rally during crash
                    daily_returns[d_idx, i] += 0.003 * rng.uniform(0.5, 1.5)
                elif group == 5:  # commodities mixed
                    daily_returns[d_idx, i] -= 0.005 * rng.uniform(0.0, 2.0)

            elif crash_bottom < date <= recovery_end:
                # Recovery phase — strong bounce
                if group in [0, 1, 2, 3, 6]:
                    daily_returns[d_idx, i] += ann_vol * 0.04 * rng.uniform(0.3, 1.2)

    # Convert returns to prices (starting at $100 for indices, realistic for stocks)
    starting_prices = {
        "SPY": 200, "QQQ": 150, "IWM": 130, "DIA": 180,
        "AAPL": 30, "MSFT": 40, "GOOGL": 50, "AMZN": 80,
        "JPM": 90, "GS": 180, "BAC": 25, "XOM": 70, "CVX": 100,
        "TLT": 120, "IEF": 105, "SHY": 84,
        "GLD": 120, "SLV": 15,
        "IWD": 110, "IWF": 130,
        "XLF": 22, "XLE": 60, "XLK": 70, "XLV": 80,
        "JNJ": 130, "PG": 80, "KO": 42, "PEP": 110, "MTUM": 100,
    }

    all_prices = {}
    for i, ticker in enumerate(tickers):
        p0 = starting_prices.get(ticker, 100)
        cumret = np.cumprod(1 + daily_returns[:, i])
        prices = p0 * cumret
        all_prices[ticker] = pd.Series(prices, index=dates, name=ticker)

    return all_prices


# Cache — generate once per session
_PRICE_CACHE = None


def _get_cache():
    global _PRICE_CACHE
    if _PRICE_CACHE is None:
        _PRICE_CACHE = _generate_all_prices()
    return _PRICE_CACHE


def get_prices(ticker: str, start: str, end: str) -> pd.Series:
    """
    Get simulated daily prices for a single ticker.

    Drop-in replacement for yfinance when offline.
    """
    cache = _get_cache()
    ticker = ticker.upper()
    if ticker not in cache:
        raise ValueError(f"Ticker '{ticker}' not in sample data. "
                         f"Available: {list(cache.keys())}")
    prices = cache[ticker]
    mask = (prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))
    return prices[mask]


def get_multi_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Get simulated daily prices for multiple tickers.

    Returns a DataFrame with tickers as columns.
    """
    cache = _get_cache()
    series = {}
    for ticker in tickers:
        t = ticker.upper()
        if t not in cache:
            raise ValueError(f"Ticker '{t}' not in sample data. "
                             f"Available: {list(cache.keys())}")
        series[t] = cache[t]

    df = pd.DataFrame(series)
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df[mask]
