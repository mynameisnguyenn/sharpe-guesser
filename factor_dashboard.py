"""
Factor Exposure Dashboard
==========================
Your Month 3-4 capstone project. Takes a list of tickers and produces
a complete factor exposure analysis — the same analysis your risk desk
runs on every PM's book.

Usage:
    python factor_dashboard.py AAPL MSFT JPM XOM
    python factor_dashboard.py AAPL --start 2020-01-01 --end 2025-12-31
    python factor_dashboard.py AAPL MSFT --rf 0.04
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so `modules/` is importable
# (needed when running from a different working directory)
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modules.module_3_factor_models import (
    capm_regression,
    build_factor_proxies,
    multifactor_regression,
    rolling_beta,
    information_ratio,
    fetch_returns,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Factor Exposure Dashboard — analyse factor exposures like a risk desk.",
    )
    parser.add_argument("tickers", nargs="+", help="One or more stock tickers (e.g. AAPL MSFT JPM)")
    parser.add_argument("--start", default="2020-01-01", help="Start date (default: 2020-01-01)")
    parser.add_argument("--end", default="2025-12-31", help="End date (default: 2025-12-31)")
    parser.add_argument("--rf", type=float, default=0.05, help="Annual risk-free rate (default: 0.05)")
    parser.add_argument("--window", type=int, default=63, help="Rolling beta window in trading days (default: 63)")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# SINGLE-TICKER ANALYSIS
# ---------------------------------------------------------------------------

def analyse_ticker(ticker, spy_returns, factors, start, end, rf, window):
    """
    Run full factor analysis for one ticker.

    Returns a summary dict (for the comparison table) or None if data fetch fails.
    This is the same workflow a risk associate runs each morning — CAPM beta,
    factor loadings, and information ratio tell you how the book is positioned.
    """
    try:
        stock_rets = fetch_returns(ticker, start, end)
    except Exception as e:
        print(f"  WARNING: Could not fetch data for {ticker}: {e}")
        return None

    if stock_rets.empty or len(stock_rets) < window:
        print(f"  WARNING: Insufficient data for {ticker}, skipping.")
        return None

    # --- CAPM ---
    capm = capm_regression(stock_rets, spy_returns, annual_rf=rf)
    print(f"\n  CAPM: {ticker} vs SPY")
    print(f"  {'-'*45}")
    print(f"    Alpha (annual) : {capm['alpha_annual']:>8.2%}  (t={capm['alpha_tstat']:.2f})")
    print(f"    Beta           : {capm['beta']:>8.3f}  (t={capm['beta_tstat']:.2f})")
    print(f"    R-squared      : {capm['r_squared']:>8.1%}")
    sig_label = "Significant alpha" if abs(capm["alpha_tstat"]) > 2 else "No significant alpha"
    print(f"    {sig_label}")

    # --- Fama-French 3-Factor ---
    mf = multifactor_regression(stock_rets, factors, annual_rf=rf)
    print(f"\n  Fama-French 3-Factor: {ticker}")
    print(f"  {'-'*50}")
    print(f"    Alpha (annual) : {mf['alpha_annual']:>8.2%}")
    print(f"    R-squared      : {mf['r_squared']:>8.1%}")
    print(f"  Factor Loadings:")
    for factor_name in ["MKT", "SMB", "HML"]:
        beta_val = mf.get(f"beta_{factor_name}", 0)
        tstat_val = mf.get(f"tstat_{factor_name}", 0)
        sig = "*" if abs(tstat_val) > 2 else ""
        print(f"    {factor_name:<6} : {beta_val:>8.3f}  (t={tstat_val:.2f}) {sig}")

    # --- Information Ratio ---
    ir = information_ratio(mf)
    print(f"\n    Information Ratio: {ir:.2f}")

    # --- Multi-panel chart ---
    rb = rolling_beta(stock_rets, spy_returns, window=window)
    residuals = mf["model"].resid

    # Cumulative returns for stock vs SPY (aligned)
    aligned = pd.concat([stock_rets, spy_returns], axis=1).dropna()
    aligned.columns = [ticker, "SPY"]
    cum_stock = (1 + aligned[ticker]).cumprod()
    cum_spy = (1 + aligned["SPY"]).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Panel 1: Rolling beta
    axes[0].plot(rb, linewidth=0.8, color="steelblue")
    axes[0].axhline(1.0, color="grey", linewidth=0.5, linestyle="--", label="Beta = 1")
    axes[0].axhline(rb.mean(), color="orange", linewidth=0.8, linestyle="--",
                    label=f"Mean = {rb.mean():.2f}")
    axes[0].set_title(f"{ticker} — {window}-day Rolling Beta to SPY")
    axes[0].set_ylabel("Beta")
    axes[0].legend()

    # Panel 2: Cumulative returns normalised to $1
    axes[1].plot(cum_stock, linewidth=0.8, label=ticker)
    axes[1].plot(cum_spy, linewidth=0.8, label="SPY", color="grey")
    axes[1].set_title(f"{ticker} vs SPY — Cumulative Returns (normalised to $1)")
    axes[1].set_ylabel("Growth of $1")
    axes[1].legend()

    # Panel 3: Residuals histogram
    axes[2].hist(residuals, bins=80, density=True, alpha=0.6, color="steelblue")
    axes[2].axvline(0, color="red", linewidth=0.5)
    axes[2].set_title(f"{ticker} — Regression Residuals Distribution")
    axes[2].set_xlabel("Residual")
    axes[2].set_ylabel("Density")

    plt.tight_layout()
    plt.close()

    return {
        "ticker": ticker,
        "alpha": capm["alpha_annual"],
        "beta": capm["beta"],
        "r_squared": capm["r_squared"],
        "smb": mf.get("beta_SMB", 0),
        "hml": mf.get("beta_HML", 0),
        "ir": ir,
        "rolling_beta": rb,
    }


# ---------------------------------------------------------------------------
# COMPARISON TABLE & CHART
# ---------------------------------------------------------------------------

def print_comparison_table(summaries):
    """Print a side-by-side comparison — the view your PM sees in the morning meeting."""
    print(f"\n{'='*70}")
    print("  FACTOR EXPOSURE COMPARISON")
    print(f"{'='*70}")
    header = f"  {'Ticker':<8} {'Alpha':>8} {'Beta':>8} {'R2':>8} {'SMB':>8} {'HML':>8} {'IR':>8}"
    print(header)
    print(f"  {'-'*62}")
    for s in summaries:
        print(f"  {s['ticker']:<8} {s['alpha']:>7.1%} {s['beta']:>8.2f} {s['r_squared']:>7.0%}"
              f" {s['smb']:>8.2f} {s['hml']:>8.2f} {s['ir']:>8.2f}")


def save_comparison_chart(summaries, window):
    """Overlay all rolling betas on one chart — shows relative risk positioning."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for s in summaries:
        ax.plot(s["rolling_beta"], linewidth=0.8, label=s["ticker"])
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--", label="Beta = 1")
    ax.set_title(f"Rolling {window}-day Beta Comparison")
    ax.set_ylabel("Beta")
    ax.legend()
    plt.tight_layout()
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    tickers = [t.upper() for t in args.tickers]

    print("\n" + "=" * 60)
    print("  FACTOR EXPOSURE DASHBOARD")
    print("=" * 60)
    print(f"  Tickers : {', '.join(tickers)}")
    print(f"  Period  : {args.start} to {args.end}")
    print(f"  Rf      : {args.rf:.1%}")
    print(f"  Window  : {args.window} days")

    # Fetch common data once — SPY and factor proxies
    print("\n  Fetching market data...")
    try:
        spy_returns = fetch_returns("SPY", args.start, args.end)
        factors = build_factor_proxies(args.start, args.end)
    except Exception as e:
        print(f"  FATAL: Could not fetch market/factor data: {e}")
        sys.exit(1)

    # Analyse each ticker
    summaries = []
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"  Analysing: {ticker}  ({args.start} to {args.end})")
        print(f"{'='*60}")

        result = analyse_ticker(ticker, spy_returns, factors,
                                args.start, args.end, args.rf, args.window)
        if result is not None:
            summaries.append(result)

    # Comparison output (only if multiple tickers succeeded)
    if len(summaries) > 1:
        print_comparison_table(summaries)
        save_comparison_chart(summaries, args.window)

    print(f"\n{'='*60}")
    print("  Dashboard complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
