"""
Data Pipeline: Fetch SPY, VIX, and Fama-French daily factors
=============================================================

Downloads and caches daily data for regime detection:
    1. SPY daily prices (2005-2024) — regime classification target
    2. VIX daily (^VIX) — implied vol feature for HMM
    3. Fama-French daily factors (MKT-RF, SMB, HML, Mom) — factor analysis

Start date is 2005 to capture the 2008 GFC regime.

Run:
    python -m data.fetch_data
"""

import io
import re
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


DATA_DIR = Path(__file__).parent


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

    print("Downloading SPY daily prices...")
    spy_raw = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy = spy_raw[["Adj Close"]].rename(columns={"Adj Close": "SPY"})
    print(f"  SPY: {len(spy)} trading days")

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
# FAMA-FRENCH DAILY FACTORS
# ---------------------------------------------------------------------------

def fetch_ff_daily(
    start: str = "2005-01-01",
    end: str = "2024-12-31",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily Fama-French 3-factor + Momentum data from Ken French's library.

    Returns DataFrame with columns: Mkt-RF, SMB, HML, RF (decimal returns).
    """
    cache_file = "ff_daily.parquet"
    if cache:
        cached = load_cached(cache_file)
        if cached is not None:
            return cached

    # FF 3 factors daily
    url_3f = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    )
    print("Downloading daily Fama-French 3-factor data...")
    req = urllib.request.Request(url_3f, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        zip_data = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            raw = f.read().decode("utf-8")

    # Parse daily rows — format: YYYYMMDD, num, num, num, num
    lines = raw.split("\n")
    data_rows = []
    for line in lines:
        line = line.strip()
        if re.match(r"^\d{8}\s*,", line):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                data_rows.append(parts[:5])

    df = pd.DataFrame(data_rows, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for col in ["Mkt-RF", "SMB", "HML", "RF"]:
        df[col] = pd.to_numeric(df[col]) / 100  # percent to decimal

    df = df.set_index("date")
    df = df.loc[start:end]

    if cache:
        cache_data(df, cache_file)

    print(f"FF daily factors: {len(df)} days")
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Regime Detection — Data Pipeline")
    print("=" * 60)
    print()

    spy, vix = fetch_spy_vix()
    ff = fetch_ff_daily()

    print()
    print("--- Summary ---")
    print(f"SPY prices:  {len(spy)} days  ({spy.index[0].date()} to {spy.index[-1].date()})")
    print(f"VIX levels:  {len(vix)} days  ({vix.index[0].date()} to {vix.index[-1].date()})")
    print(f"FF factors:  {len(ff)} days  ({ff.index[0].date()} to {ff.index[-1].date()})")
    print()
    print("Data pipeline complete.")
