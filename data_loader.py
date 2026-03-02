"""
Data Loader — Tries Yahoo Finance, Falls Back to Sample Data
=============================================================

All modules import from here instead of yfinance directly.
This way the learning scripts work both online and offline.

Usage:
    from data_loader import fetch_prices, fetch_multi_prices
"""

import pandas as pd


def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    """
    Fetch adjusted close prices for a single ticker.

    Tries Yahoo Finance first; falls back to synthetic sample data
    if the download fails (no internet, proxy block, etc.).
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        prices = df["Close"].squeeze()
        if prices.empty or prices.isna().all():
            raise ValueError("Empty data")
        return prices
    except Exception:
        from sample_data import get_prices
        print(f"  [offline mode] Using sample data for {ticker}")
        return get_prices(ticker, start, end)


def fetch_multi_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices for multiple tickers.

    Tries Yahoo Finance first; falls back to synthetic sample data.
    """
    try:
        import yfinance as yf
        df = yf.download(tickers, start=start, end=end, progress=False)
        prices = df["Close"]
        if prices.empty or prices.isna().all().all():
            raise ValueError("Empty data")
        return prices
    except Exception:
        from sample_data import get_multi_prices
        print(f"  [offline mode] Using sample data for {tickers}")
        return get_multi_prices(tickers, start, end)
