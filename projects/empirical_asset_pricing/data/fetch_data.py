"""
Data Pipeline: Fetch S&P 500 stock data via yfinance
=====================================================

This module handles the unglamorous but critical first step: getting clean data.
On a risk desk you rely on Bloomberg or a vendor feed. Here we use yfinance
(free Yahoo Finance wrapper) to pull daily OHLCV data for the S&P 500 universe.

Pipeline:
    1. Scrape the current S&P 500 ticker list from Wikipedia
    2. Download daily OHLCV data via yfinance (with error handling for delisted tickers)
    3. Resample to monthly returns (the prediction horizon in the Kelly paper)
    4. Cache everything to parquet so we only hit the API once

Run:
    python -m data.fetch_data
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# All cached data lives next to this file
DATA_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# 1. GET THE S&P 500 UNIVERSE
# ---------------------------------------------------------------------------
# The Kelly paper uses the full CRSP universe (~30,000 stocks). We simplify
# to the current S&P 500 — large, liquid names with good data coverage.
# Survivorship bias caveat: we're using today's index constituents, which
# means we only include stocks that survived. A production system would
# track historical index membership. For a portfolio project, this is fine.

def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    # The first table on the page has the ticker list
    sp500_table = tables[0]
    tickers = sp500_table["Symbol"].tolist()

    # Clean up tickers — Wikipedia sometimes uses dots (BRK.B) where
    # yfinance expects dashes (BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(tickers)


# ---------------------------------------------------------------------------
# 2. DOWNLOAD DAILY OHLCV DATA
# ---------------------------------------------------------------------------
# yfinance is free but unreliable for large batch downloads. We download
# one ticker at a time with error handling so one bad ticker doesn't kill
# the entire pipeline. This is slower but more robust.

def fetch_stock_data(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    pause: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV data for a list of tickers.

    Returns a dict of {ticker: DataFrame} where each DataFrame has columns
    [Open, High, Low, Close, Adj Close, Volume]. Tickers that fail to
    download are silently skipped (logged to stdout).
    """
    stock_data = {}
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
            )
            if df.empty:
                failed.append(ticker)
                continue

            # yfinance sometimes returns MultiIndex columns for single tickers
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            stock_data[ticker] = df

            if (i + 1) % 50 == 0:
                print(f"  Downloaded {i + 1}/{len(tickers)} tickers...")

        except Exception as e:
            failed.append(ticker)
            print(f"  Failed to download {ticker}: {e}")

        # Small pause to avoid hammering Yahoo Finance
        time.sleep(pause)

    print(f"\nDownloaded {len(stock_data)} tickers, {len(failed)} failed.")
    if failed:
        print(f"Failed tickers: {failed[:20]}{'...' if len(failed) > 20 else ''}")

    return stock_data


# ---------------------------------------------------------------------------
# 3. COMPUTE MONTHLY RETURNS
# ---------------------------------------------------------------------------
# The Kelly paper predicts monthly returns. We resample daily adjusted close
# prices to month-end, then compute simple returns. Using adjusted close
# accounts for dividends and splits — important for getting returns right.

def compute_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame of daily adjusted close prices to monthly returns.

    Input:  DataFrame with DatetimeIndex, columns = tickers, values = adj close
    Output: DataFrame with monthly PeriodIndex, columns = tickers, values = returns
    """
    # Resample to month-end prices
    monthly_prices = daily_prices.resample("ME").last()

    # Simple returns: (P_t / P_{t-1}) - 1
    monthly_returns = monthly_prices.pct_change()

    # Drop the first row (NaN from pct_change)
    monthly_returns = monthly_returns.iloc[1:]

    return monthly_returns


# ---------------------------------------------------------------------------
# 4. CACHING — SAVE AND LOAD FROM DISK
# ---------------------------------------------------------------------------
# Downloading 500 stocks takes several minutes. We cache to parquet (fast,
# columnar format) so subsequent runs load instantly. pyarrow is the engine.

def cache_data(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame to parquet in the data/ directory."""
    filepath = DATA_DIR / filename
    df.to_parquet(filepath, engine="pyarrow")
    print(f"Cached: {filepath} ({len(df)} rows, {len(df.columns)} cols)")
    return filepath


def load_cached(filename: str) -> Optional[pd.DataFrame]:
    """Load a cached DataFrame from parquet. Returns None if not found."""
    filepath = DATA_DIR / filename
    if filepath.exists():
        df = pd.read_parquet(filepath, engine="pyarrow")
        print(f"Loaded from cache: {filepath} ({len(df)} rows, {len(df.columns)} cols)")
        return df
    return None


# ---------------------------------------------------------------------------
# 5. BUILD THE DATASET — ORCHESTRATE EVERYTHING
# ---------------------------------------------------------------------------

def build_dataset(
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: fetch S&P 500 data, compute monthly returns, cache results.

    Returns:
        prices:   DataFrame of daily adjusted close prices (tickers as columns)
        volumes:  DataFrame of daily volumes (tickers as columns)
        returns:  DataFrame of monthly returns (tickers as columns)
    """
    # Check cache first
    if cache:
        prices = load_cached("daily_prices.parquet")
        volumes = load_cached("daily_volumes.parquet")
        returns = load_cached("monthly_returns.parquet")
        if prices is not None and volumes is not None and returns is not None:
            return prices, volumes, returns

    # Step 1: Get tickers
    print("Fetching S&P 500 ticker list...")
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers.\n")

    # Step 2: Download daily data
    print("Downloading daily data (this takes a few minutes)...")
    stock_data = fetch_stock_data(tickers, start=start, end=end)

    # Step 3: Assemble into price and volume matrices
    # Each column is a ticker, each row is a date
    prices = pd.DataFrame(
        {ticker: df["Adj Close"] for ticker, df in stock_data.items()}
    )
    volumes = pd.DataFrame(
        {ticker: df["Volume"] for ticker, df in stock_data.items()}
    )

    # Step 4: Monthly returns
    returns = compute_monthly_returns(prices)

    # Step 5: Cache
    if cache:
        cache_data(prices, "daily_prices.parquet")
        cache_data(volumes, "daily_volumes.parquet")
        cache_data(returns, "monthly_returns.parquet")

    return prices, volumes, returns


# ---------------------------------------------------------------------------
# MAIN — run the pipeline directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Empirical Asset Pricing — Data Pipeline")
    print("=" * 60)
    print()

    prices, volumes, returns = build_dataset()

    print()
    print("--- Summary ---")
    print(f"Daily prices:    {prices.shape[0]} days x {prices.shape[1]} stocks")
    print(f"Daily volumes:   {volumes.shape[0]} days x {volumes.shape[1]} stocks")
    print(f"Monthly returns: {returns.shape[0]} months x {returns.shape[1]} stocks")
    print(f"Date range:      {prices.index[0].date()} to {prices.index[-1].date()}")
    print()
    print("Data pipeline complete. Cached to data/ directory.")
