"""
Volatility Forecasting — End-to-End Pipeline
==============================================

Compares four volatility forecasting models on SPY daily returns (2005-2024):
    1. EWMA (RiskMetrics, lambda=0.94) — simplest baseline
    2. GARCH(1,1) — arch package, expanding window
    3. HAR-RV (Corsi 2009) — OLS of RV on daily/weekly/monthly lagged RV
    4. VIX implied — the market's own forecast

Evaluation:
    - QLIKE, MSE, MAE loss functions
    - Mincer-Zarnowitz unbiasedness regression
    - Time series and scatter plots

Vol-targeting backtest (Paleologo Ch 6):
    - Equal-weight portfolio of 50 liquid S&P stocks
    - Scale position size so predicted vol = constant target
    - Compare Sharpe, drawdown, kurtosis vs unmanaged portfolio

Run from the project directory:
    python run_pipeline.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetch_data import fetch_spy_vix, fetch_universe
from src.realized_vol import realized_vol, forward_rv, har_features, vol_cone
from src.models import ewma_vol, garch_vol, har_rv_vol, vix_implied_vol, run_all_forecasts
from src.evaluate import (
    compare_models,
    plot_forecast_vs_realized,
    plot_mz_scatter,
    plot_vol_cone,
    plot_vol_target_comparison,
)
from src.vol_target import (
    equal_weight_returns,
    vol_targeted_returns,
    run_vol_target_backtest,
)

RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. FETCH DATA
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Step 1: Fetching data")
    print("=" * 60)

    spy_df, vix_df = fetch_spy_vix()
    # Squeeze DataFrames to Series for model functions
    spy_prices = spy_df.squeeze()
    vix = vix_df.squeeze()
    print(f"  SPY: {len(spy_prices)} trading days "
          f"({spy_prices.index[0].date()} to {spy_prices.index[-1].date()})")
    print(f"  VIX: {len(vix)} trading days")

    universe_prices = fetch_universe()
    print(f"  Universe: {universe_prices.shape[1]} stocks, {universe_prices.shape[0]} days")

    # Compute SPY daily returns
    spy_returns = spy_prices.pct_change().dropna()
    spy_returns.name = "SPY"
    print(f"  SPY daily returns: {len(spy_returns)} observations")

    # ------------------------------------------------------------------
    # 2. REALIZED VOL & FEATURES
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 2: Computing realized vol and features")
    print("=" * 60)

    rv = realized_vol(spy_returns, window=22)
    fwd_rv = forward_rv(spy_returns, window=22)
    features = har_features(spy_returns)

    print(f"  Realized vol (22d): {rv.dropna().shape[0]} observations")
    print(f"  Forward RV target:  {fwd_rv.dropna().shape[0]} observations")
    print(f"  HAR features:       {features.dropna().shape[0]} observations")
    print(f"  Current RV:         {rv.dropna().iloc[-1]:.1%}")

    # Vol cone
    cone = vol_cone(spy_returns)
    print(f"\n  Volatility Cone:")
    print(cone.to_string())

    # Current RV at each horizon for overlay
    current_rv = {}
    for w in [5, 10, 22, 63, 126, 252]:
        rv_w = realized_vol(spy_returns, window=w)
        current_rv[w] = rv_w.dropna().iloc[-1]

    plot_vol_cone(
        cone,
        current_rv=current_rv,
        title="SPY Volatility Cone (2005-2024)",
        save_path=str(RESULTS_DIR / "vol_cone.png"),
    )

    # ------------------------------------------------------------------
    # 3. RUN ALL FORECASTS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 3: Running vol forecast models")
    print("=" * 60)

    print("  Running EWMA...")
    ewma = ewma_vol(spy_returns)
    print(f"    EWMA: {ewma.dropna().shape[0]} forecasts")

    print("  Running GARCH(1,1)...")
    garch = garch_vol(spy_returns, refit_every=22, min_history=252)
    print(f"    GARCH: {garch.dropna().shape[0]} forecasts")

    print("  Running HAR-RV...")
    har = har_rv_vol(spy_returns, min_history=252, refit_every=22)
    print(f"    HAR-RV: {har.dropna().shape[0]} forecasts")

    print("  Aligning VIX implied...")
    vix_vol = vix_implied_vol(vix)
    print(f"    VIX: {vix_vol.dropna().shape[0]} forecasts")

    # Combine into DataFrame
    forecasts = run_all_forecasts(spy_returns, vix)
    print(f"\n  Combined forecast DataFrame: {len(forecasts)} rows, "
          f"{len(forecasts.columns)} columns")
    print(f"  Date range: {forecasts.index[0].date()} to {forecasts.index[-1].date()}")

    # ------------------------------------------------------------------
    # 4. EVALUATE FORECASTS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 4: Evaluating forecast accuracy")
    print("=" * 60)

    comparison = compare_models(forecasts)

    # Forecast vs realized time series
    plot_forecast_vs_realized(
        forecasts,
        title="SPY Volatility: Forecasts vs Realized (22-day)",
        save_path=str(RESULTS_DIR / "forecast_vs_realized.png"),
    )

    # MZ scatter plots
    plot_mz_scatter(
        forecasts,
        save_path=str(RESULTS_DIR / "mz_scatter.png"),
    )

    # ------------------------------------------------------------------
    # 5. VOL-TARGETING BACKTEST
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 5: Vol-targeting backtest (Paleologo Ch 6)")
    print("=" * 60)

    # Use EWMA vol on the equal-weight portfolio itself for targeting
    portfolio_returns = equal_weight_returns(universe_prices)
    portfolio_ewma = ewma_vol(portfolio_returns)

    print(f"  Portfolio returns: {len(portfolio_returns)} days")
    print(f"  Portfolio EWMA vol: {portfolio_ewma.dropna().shape[0]} days")

    results = run_vol_target_backtest(
        universe_prices,
        vol_forecast=portfolio_ewma,
        target_vol=0.10,
    )

    plot_vol_target_comparison(
        results,
        save_path=str(RESULTS_DIR / "vol_target_comparison.png"),
    )

    # ------------------------------------------------------------------
    # 6. SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    print("\n  Forecast Model Rankings (by QLIKE, lower is better):")
    ranked = comparison.sort_values("QLIKE")
    for i, (model, row) in enumerate(ranked.iterrows(), 1):
        print(f"    {i}. {model.upper():8s}  QLIKE={row['QLIKE']:.4f}  "
              f"MZ R²={row['MZ_R2']:.1%}  MAE={row['MAE']:.4f}")

    print(f"\n  Vol-Targeting Results:")
    um = results["unmanaged"]["metrics"]
    mm = results["managed"]["metrics"]
    print(f"    Unmanaged Sharpe: {um['sharpe']:.3f}  |  Managed Sharpe: {mm['sharpe']:.3f}")
    print(f"    Unmanaged Max DD: {um['max_drawdown']:.1%}  |  "
          f"Managed Max DD: {mm['max_drawdown']:.1%}")
    print(f"    Sharpe improvement: {mm['sharpe'] - um['sharpe']:+.3f}")

    print(f"\n  Results saved to: {RESULTS_DIR}")
    print("    - vol_cone.png")
    print("    - forecast_vs_realized.png")
    print("    - mz_scatter.png")
    print("    - vol_target_comparison.png")
    print("\n  Pipeline complete.")


if __name__ == "__main__":
    main()
