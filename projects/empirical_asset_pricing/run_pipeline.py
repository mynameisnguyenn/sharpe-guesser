"""
Empirical Asset Pricing via Machine Learning — End-to-End Pipeline
===================================================================

Replicates the core analysis from Gu, Kelly, Xiu (2020) using a simplified
feature set and two ML models (elastic net and random forest). The pipeline:

    1. Fetch S&P 500 daily data (or load from cache)
    2. Build firm-level features (momentum, volatility, size, liquidity)
    3. Construct monthly prediction targets (next-month return)
    4. Walk-forward predict with elastic net and random forest
    5. Form long-short decile portfolios from predictions
    6. Compare strategies and visualize results

Run from the project directory:
    python run_pipeline.py
"""

import matplotlib
matplotlib.use("Agg")

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is on the path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetch_data import build_dataset
from src.features import build_features
from src.models import (
    train_elastic_net,
    predict_elastic_net,
    train_random_forest,
    expanding_window_predict,
)
from src.portfolio import rank_stocks, long_short_returns, compute_performance, quantile_returns
from src.evaluate import (
    compare_strategies,
    feature_importance_plot,
    plot_cumulative,
    plot_decile_returns,
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

    prices, volumes, returns = build_dataset()

    print(f"\nDaily prices:    {prices.shape[0]} days x {prices.shape[1]} stocks")
    print(f"Daily volumes:   {volumes.shape[0]} days x {volumes.shape[1]} stocks")
    print(f"Monthly returns: {returns.shape[0]} months x {returns.shape[1]} stocks")

    # ------------------------------------------------------------------
    # 2. BUILD FEATURES
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 2: Building features")
    print("=" * 60)

    # build_features expects daily prices and volumes, returns daily features
    # in long format (date, ticker) MultiIndex
    daily_returns = prices.pct_change()
    features_df = build_features(prices, volumes, returns=daily_returns)

    print(f"Daily features: {len(features_df)} rows, {len(features_df.columns)} features")
    print(f"Feature columns: {list(features_df.columns)}")
    feature_cols = list(features_df.columns)

    # ------------------------------------------------------------------
    # 3. RESAMPLE FEATURES TO MONTHLY & CREATE TARGET
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 3: Resampling features to monthly & creating target")
    print("=" * 60)

    # Features are daily — resample to month-end (take last value each month)
    features_df = features_df.reset_index()
    features_df["month_end"] = features_df["date"].dt.to_period("M")
    # For each (ticker, month), take the last daily observation
    monthly_features = (
        features_df
        .sort_values("date")
        .groupby(["ticker", "month_end"])
        .last()
        .reset_index()
    )
    # Convert month_end Period back to Timestamp (end of month) for alignment
    monthly_features["date"] = monthly_features["month_end"].dt.to_timestamp(how="end")
    monthly_features = monthly_features.drop(columns=["month_end"])

    # Stack monthly returns from wide (dates x tickers) to long (date, ticker)
    monthly_ret_long = returns.stack()
    monthly_ret_long.name = "monthly_return"
    monthly_ret_long.index.names = ["date", "ticker"]
    monthly_ret_long = monthly_ret_long.reset_index()

    # Align dates: monthly returns use month-end dates from resample("ME"),
    # so normalize both to month-end
    monthly_features["date"] = monthly_features["date"].dt.to_period("M")
    monthly_ret_long["date"] = monthly_ret_long["date"].dt.to_period("M")

    # Merge features with monthly returns
    data = monthly_features.merge(monthly_ret_long, on=["date", "ticker"], how="inner")

    # Create next-month return target: for each ticker, shift return by -1
    data = data.sort_values(["ticker", "date"])
    data["next_month_return"] = data.groupby("ticker")["monthly_return"].shift(-1)

    # Drop rows without a target (last month for each ticker)
    data = data.dropna(subset=["next_month_return"])

    # Convert period back to timestamp for the model (expanding_window_predict
    # needs sortable dates)
    data["date"] = data["date"].dt.to_timestamp()

    # Set (date, ticker) as MultiIndex — expanding_window_predict expects this
    data = data.set_index(["date", "ticker"])

    print(f"Model dataset: {len(data)} rows")
    print(f"Date range: {data.index.get_level_values(0).min().date()} to "
          f"{data.index.get_level_values(0).max().date()}")
    print(f"Unique months: {data.index.get_level_values(0).nunique()}")
    print(f"Features: {feature_cols}")

    # ------------------------------------------------------------------
    # 4. EXPANDING WINDOW PREDICTIONS
    # ------------------------------------------------------------------
    target = "next_month_return"
    min_periods = 36

    # --- Elastic Net ---
    print("\n" + "=" * 60)
    print("  Step 4a: Expanding window prediction — Elastic Net")
    print("=" * 60)
    print(f"Training with min_periods={min_periods} (first {min_periods} months for training)...")

    enet_preds = expanding_window_predict(
        data=data,
        features=feature_cols,
        target=target,
        model_fn=train_elastic_net,
        predict_fn=predict_elastic_net,
        min_periods=min_periods,
    )
    print(f"Elastic Net predictions: {len(enet_preds)} stock-months")

    # --- Random Forest ---
    print("\n" + "=" * 60)
    print("  Step 4b: Expanding window prediction — Random Forest")
    print("=" * 60)
    print(f"Training with min_periods={min_periods}...")

    rf_preds = expanding_window_predict(
        data=data,
        features=feature_cols,
        target=target,
        model_fn=train_random_forest,
        predict_fn=None,
        min_periods=min_periods,
    )
    print(f"Random Forest predictions: {len(rf_preds)} stock-months")

    # ------------------------------------------------------------------
    # 5. PORTFOLIO CONSTRUCTION
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 5: Building long-short portfolios")
    print("=" * 60)

    # Elastic Net portfolio
    enet_rankings = rank_stocks(enet_preds["prediction"])
    enet_ls = long_short_returns(enet_preds["actual"], enet_rankings)
    print(f"Elastic Net long-short: {len(enet_ls)} months")

    # Random Forest portfolio
    rf_rankings = rank_stocks(rf_preds["prediction"])
    rf_ls = long_short_returns(rf_preds["actual"], rf_rankings)
    print(f"Random Forest long-short: {len(rf_ls)} months")

    # ------------------------------------------------------------------
    # 6. EVALUATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 6: Evaluating strategies")
    print("=" * 60)

    strategy_dict = {
        "Elastic Net L/S": enet_ls,
        "Random Forest L/S": rf_ls,
    }

    # Strategy comparison table
    comparison = compare_strategies(strategy_dict)

    # Cumulative returns plot
    plot_cumulative(
        strategy_dict,
        title="Cumulative Returns: ML Long-Short Portfolios",
        save_path=str(RESULTS_DIR / "cumulative_returns.png"),
    )

    # Feature importance from random forest
    # Re-train a final RF model on all data to get stable feature importances
    print("Computing feature importance from Random Forest...")
    all_X = data[feature_cols].dropna()
    all_y = data.loc[all_X.index, target].dropna()
    common_idx = all_X.index.intersection(all_y.index)
    all_X = all_X.loc[common_idx]
    all_y = all_y.loc[common_idx]
    final_rf = train_random_forest(all_X, all_y)

    feature_importance_plot(
        final_rf,
        feature_cols,
        save_path=str(RESULTS_DIR / "feature_importance.png"),
    )

    # Decile return analysis for each model
    print("\nDecile return analysis...")

    enet_qr = quantile_returns(enet_preds["actual"], enet_rankings)
    print("\nElastic Net — Returns by Prediction Decile:")
    print(enet_qr.to_string())

    plot_decile_returns(
        enet_qr,
        title="Elastic Net: Average Returns by Prediction Decile",
        save_path=str(RESULTS_DIR / "decile_returns_enet.png"),
    )

    rf_qr = quantile_returns(rf_preds["actual"], rf_rankings)
    print("\nRandom Forest — Returns by Prediction Decile:")
    print(rf_qr.to_string())

    plot_decile_returns(
        rf_qr,
        title="Random Forest: Average Returns by Prediction Decile",
        save_path=str(RESULTS_DIR / "decile_returns_rf.png"),
    )

    # ------------------------------------------------------------------
    # 7. SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Summary of Findings")
    print("=" * 60)

    for name, ls_returns in strategy_dict.items():
        perf = compute_performance(ls_returns)
        print(f"\n{name}:")
        print(f"  Annualized Return: {perf['annual_return']:.2%}")
        print(f"  Annualized Vol:    {perf['annual_vol']:.2%}")
        print(f"  Sharpe Ratio:      {perf['sharpe']:.3f}")
        print(f"  Sortino Ratio:     {perf['sortino']:.3f}")
        print(f"  Max Drawdown:      {perf['max_drawdown']:.2%}")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("  - cumulative_returns.png")
    print("  - feature_importance.png")
    print("  - decile_returns_enet.png")
    print("  - decile_returns_rf.png")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
