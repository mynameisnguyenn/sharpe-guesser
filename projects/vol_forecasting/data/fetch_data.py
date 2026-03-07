"""
Data Pipeline: Fetch SPY, VIX, and liquid stock universe via yfinance
=====================================================================

Downloads and caches daily price data for vol forecasting:
    1. SPY daily prices (2005-2024) — the primary forecast target
    2. VIX daily (^VIX) — implied vol benchmark
    3. 50 liquid S&P 500 stocks — for the vol-targeting backtest

Start date is 2005 to capture the 2008 GFC regime.

Run:
    python -m data.fetch_data
"""

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


# All cached data lives next to this file
DATA_DIR = Path(__file__).parent

# 50 liquid, well-known S&P 500 names with good data coverage back to 2005
LIQUID_50 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "ADBE", "CRM",
    "NFLX", "CMCSA", "PEP", "TMO", "COST", "AVGO", "CSCO", "ABT", "ACN",
    "MRK", "NKE", "WMT", "DHR", "LLY", "TXN", "MDT", "NEE", "PM", "HON",
    "UNP", "AMGN", "INTC", "LOW", "QCOM", "BA", "CAT", "GS", "BLK",
    "SBUX", "MMM", "GE", "CVX",
]


# ---------------------------------------------------------------------------
# CACHING UTILITIES
# ---------------------------------------------------------------------------

def cache_data(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame to parquet in the data/ directory."""
    filepath = DATA_DIR / filename
    df.to_parquet(filepath, engine="pyarrow")
    print(f"Cached: {filepath.name} ({len(df)} rows)")
    return filepath


def load_cached(filename: str) -> Optional[pd.DataFrame]:
    """Load a cached DataFrame from parquet. Returns None if not found."""
    filepath = DATA_DIR / filename
    if filepath.exists():
        df = pd.read_parquet(filepath, engine="pyarrow")
        print(f"Loaded from cache: {filepath.name} ({len(df)} rows)")
        return df
    return None


# ---------------------------------------------------------------------------
# SPY + VIX
# ---------------------------------------------------------------------------
# SPY is the forecast target — we predict its future realized vol.
# VIX is the market's implied vol estimate — our benchmark to beat.
# VIX values are in percentage points (e.g., 20.5 = 20.5% annualized vol).

def fetch_spy_vix(
    start: str = "2005-01-01",
    end: str = "2024-12-31",
    cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch SPY adj close prices and VIX close. Return (spy_prices, vix)."""
    spy_cache = "spy_prices.parquet"
    vix_cache = "vix.parquet"

    if cache:
        spy = load_cached(spy_cache)
        vix = load_cached(vix_cache)
        if spy is not None and vix is not None:
            return spy, vix

    # --- SPY ---
    print("Downloading SPY daily prices...")
    spy_raw = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy = spy_raw[["Adj Close"]].rename(columns={"Adj Close": "SPY"})
    print(f"  SPY: {len(spy)} trading days")

    # --- VIX ---
    print("Downloading VIX daily levels...")
    vix_raw = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    vix = vix_raw[["Close"]].rename(columns={"Close": "VIX"})
    print(f"  VIX: {len(vix)} trading days")

    if cache:
        cache_data(spy, spy_cache)
        cache_data(vix, vix_cache)

    return spy, vix


# ---------------------------------------------------------------------------
# STOCK UNIVERSE (50 liquid names)
# ---------------------------------------------------------------------------
# Used for the vol-targeting backtest. We download adj close prices for each
# ticker individually to handle failures gracefully.

def fetch_universe(
    tickers: list[str] = None,
    start: str = "2005-01-01",
    end: str = "2024-12-31",
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily adj close prices for 50 liquid stocks. Return wide DataFrame."""
    if tickers is None:
        tickers = LIQUID_50

    universe_cache = "universe_prices.parquet"
    if cache:
        cached = load_cached(universe_cache)
        if cached is not None:
            return cached

    print(f"Downloading {len(tickers)} stocks...")
    prices = {}
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                progress=False, auto_adjust=False,
            )
            if df.empty:
                failed.append(ticker)
                continue

            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            prices[ticker] = df["Adj Close"]

            if (i + 1) % 10 == 0:
                print(f"  Downloaded {i + 1}/{len(tickers)} tickers...")

        except Exception as e:
            failed.append(ticker)
            print(f"  Failed: {ticker} — {e}")

        time.sleep(0.1)  # small pause to avoid hammering Yahoo Finance

    universe = pd.DataFrame(prices)
    print(f"\nDownloaded {len(prices)} tickers, {len(failed)} failed.")
    if failed:
        print(f"Failed tickers: {failed}")

    if cache:
        cache_data(universe, universe_cache)

    return universe


# ---------------------------------------------------------------------------
# MAIN — run the full fetch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Vol Forecasting — Data Pipeline")
    print("=" * 60)
    print()

    spy, vix = fetch_spy_vix()
    universe = fetch_universe()

    print()
    print("--- Summary ---")
    print(f"SPY prices:  {len(spy)} days  ({spy.index[0].date()} to {spy.index[-1].date()})")
    print(f"VIX levels:  {len(vix)} days  ({vix.index[0].date()} to {vix.index[-1].date()})")
    print(f"Universe:    {universe.shape[0]} days x {universe.shape[1]} stocks")
    print()
    print("Data pipeline complete. Cached to data/ directory.")
