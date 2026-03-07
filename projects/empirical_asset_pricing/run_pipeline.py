"""
Empirical Asset Pricing via Machine Learning — End-to-End Pipeline
===================================================================

Replicates the core analysis from Gu, Kelly, Xiu (2020) using a simplified
feature set and three ML models (elastic net, random forest, gradient boosted
trees). The pipeline:

    1. Fetch S&P 500 daily data (or load from cache)
    2. Build firm-level features (momentum, volatility, size, liquidity)
    3. Add feature interactions (pairwise products)
    4. Construct monthly prediction targets (next-month return)
    5. Walk-forward predict with elastic net, random forest, and GBT
    6. Form long-short decile portfolios from predictions
    7. Evaluate: OOS R², Fama-French alpha, significance tests, turnover

Run from the project directory:
    python run_pipeline.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

from data.fetch_data import build_dataset, fetch_fama_french_factors
from src.features import build_features, build_interactions
from src.models import (
    train_elastic_net,
    predict_elastic_net,
    train_random_forest,
    train_gradient_boosting,
    expanding_window_predict,
)
from src.portfolio import (
    rank_stocks,
    long_short_returns,
    compute_performance,
    quantile_returns,
    compute_turnover,
    net_of_cost_returns,
)
from src.evaluate import (
    compare_strategies,
    feature_importance_plot,
    plot_cumulative,
    plot_decile_returns,
    oos_r_squared,
    fama_french_alpha,
    spread_significance,
    turnover_cost_summary,
)

RESULTS_DIR = PROJECT_ROOT / "results"
COST_BPS = 10  # One-way transaction cost assumption (basis points)


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

    # Fetch Fama-French factors for benchmark comparison
    ff_factors = fetch_fama_french_factors()

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

    print(f"Base features: {len(features_df)} rows, {len(features_df.columns)} features")
    print(f"Feature columns: {list(features_df.columns)}")

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

    # Add feature interactions (pairwise products of base features)
    base_feature_cols = [c for c in monthly_features.columns
                         if c not in ("date", "ticker")]
    interaction_df = build_interactions(
        monthly_features.set_index(["date", "ticker"])[base_feature_cols]
    )
    monthly_features = monthly_features.set_index(["date", "ticker"])
    monthly_features = pd.concat(
        [monthly_features[base_feature_cols], interaction_df.drop(columns=base_feature_cols)],
        axis=1,
    ).reset_index()

    feature_cols = [c for c in monthly_features.columns
                    if c not in ("date", "ticker")]
    print(f"Total features (base + interactions): {len(feature_cols)}")

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

    # ------------------------------------------------------------------
    # 4. EXPANDING WINDOW PREDICTIONS (cached to parquet for speed)
    # ------------------------------------------------------------------
    target = "next_month_return"
    min_periods = 36
    CACHE_DIR = PROJECT_ROOT / "data"

    def cached_predict(name, model_fn, predict_fn=None):
        """Train with expanding window, or load cached predictions."""
        cache_file = CACHE_DIR / f"preds_{name}.parquet"
        if cache_file.exists():
            preds = pd.read_parquet(cache_file)
            print(f"  Loaded cached predictions: {cache_file.name} ({len(preds)} rows)")
            return preds
        print(f"  Training {name} (this takes several minutes, cached after)...")
        preds = expanding_window_predict(
            data=data,
            features=feature_cols,
            target=target,
            model_fn=model_fn,
            predict_fn=predict_fn,
            min_periods=min_periods,
        )
        preds.to_parquet(cache_file)
        print(f"  Cached to {cache_file.name}")
        return preds

    # --- Elastic Net ---
    print("\n" + "=" * 60)
    print("  Step 4a: Expanding window prediction — Elastic Net")
    print("=" * 60)
    enet_preds = cached_predict("elastic_net", train_elastic_net, predict_elastic_net)
    print(f"Elastic Net predictions: {len(enet_preds)} stock-months")

    # --- Random Forest ---
    print("\n" + "=" * 60)
    print("  Step 4b: Expanding window prediction — Random Forest")
    print("=" * 60)
    rf_preds = cached_predict("random_forest", train_random_forest)
    print(f"Random Forest predictions: {len(rf_preds)} stock-months")

    # --- Gradient Boosted Trees ---
    print("\n" + "=" * 60)
    print("  Step 4c: Expanding window prediction — Gradient Boosting")
    print("=" * 60)
    gbt_preds = cached_predict("gradient_boosting", train_gradient_boosting)
    print(f"Gradient Boosting predictions: {len(gbt_preds)} stock-months")

    # ------------------------------------------------------------------
    # 5. OUT-OF-SAMPLE R-SQUARED
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 5: Out-of-Sample R²")
    print("=" * 60)

    for name, preds in [("Elastic Net", enet_preds),
                        ("Random Forest", rf_preds),
                        ("Gradient Boosting", gbt_preds)]:
        r2 = oos_r_squared(preds)
        print(f"  {name} OOS R²: {r2:.4f} ({r2 * 100:.2f}%)")

    # ------------------------------------------------------------------
    # 6. PORTFOLIO CONSTRUCTION
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 6: Building long-short portfolios")
    print("=" * 60)

    models = {
        "Elastic Net": enet_preds,
        "Random Forest": rf_preds,
        "Gradient Boosting": gbt_preds,
    }

    strategy_dict = {}
    rankings_dict = {}
    turnover_dict = {}
    net_returns_dict = {}

    for name, preds in models.items():
        rankings = rank_stocks(preds["prediction"])
        ls = long_short_returns(preds["actual"], rankings)
        to = compute_turnover(rankings)
        net = net_of_cost_returns(ls, to, cost_bps=COST_BPS)

        strategy_dict[f"{name} L/S"] = ls
        rankings_dict[name] = rankings
        turnover_dict[name] = to
        net_returns_dict[f"{name} L/S"] = net

        print(f"  {name}: {len(ls)} months, avg turnover {to.mean():.1%}")

    # ------------------------------------------------------------------
    # 7. EVALUATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 7: Evaluating strategies")
    print("=" * 60)

    # Strategy comparison table (gross returns)
    comparison = compare_strategies(strategy_dict)

    # Net-of-cost comparison
    print(f"\n  --- After Transaction Costs ({COST_BPS} bps one-way) ---")
    net_comparison = compare_strategies(net_returns_dict)

    # Cumulative returns plot (include net-of-cost)
    all_strategies = {}
    all_strategies.update(strategy_dict)
    for name, net in net_returns_dict.items():
        all_strategies[f"{name} (net)"] = net

    plot_cumulative(
        all_strategies,
        title="Cumulative Returns: ML Long-Short Portfolios (Gross & Net)",
        save_path=str(RESULTS_DIR / "cumulative_returns.png"),
    )

    # Feature importance from random forest
    print("\nComputing feature importance from Random Forest...")
    all_X = data[feature_cols].dropna()
    all_y = data.loc[all_X.index, target].dropna()
    common_idx = all_X.index.intersection(all_y.index)
    all_X = all_X.loc[common_idx]
    all_y = all_y.loc[common_idx]
    final_rf = train_random_forest(all_X, all_y)

    feature_importance_plot(
        final_rf,
        feature_cols,
        top_n=15,
        save_path=str(RESULTS_DIR / "feature_importance.png"),
    )

    # Feature importance from gradient boosting (for comparison)
    print("Computing feature importance from Gradient Boosting...")
    final_gbt = train_gradient_boosting(all_X, all_y)

    feature_importance_plot(
        final_gbt,
        feature_cols,
        top_n=15,
        save_path=str(RESULTS_DIR / "feature_importance_gbt.png"),
    )

    # Decile return analysis for each model
    print("\nDecile return analysis...")

    for name, preds in models.items():
        rankings = rankings_dict[name]
        qr = quantile_returns(preds["actual"], rankings)
        safe_name = name.lower().replace(" ", "_")

        print(f"\n{name} — Returns by Prediction Decile:")
        print(qr.to_string())

        plot_decile_returns(
            qr,
            title=f"{name}: Average Returns by Prediction Decile",
            save_path=str(RESULTS_DIR / f"decile_returns_{safe_name}.png"),
        )

    # ------------------------------------------------------------------
    # 8. FAMA-FRENCH ALPHA REGRESSION
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 8: Fama-French Factor Regression")
    print("=" * 60)

    for name, ls_returns in strategy_dict.items():
        fama_french_alpha(ls_returns, ff_factors, name=name)

    # ------------------------------------------------------------------
    # 9. STATISTICAL SIGNIFICANCE
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 9: Statistical Significance of L/S Spread")
    print("=" * 60)

    for name, ls_returns in strategy_dict.items():
        spread_significance(ls_returns, name=name)

    # ------------------------------------------------------------------
    # 10. TURNOVER & TRANSACTION COSTS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Step 10: Turnover & Transaction Costs")
    print("=" * 60)

    for name in models:
        ls_key = f"{name} L/S"
        turnover_cost_summary(
            strategy_dict[ls_key],
            turnover_dict[name],
            net_returns_dict[ls_key],
            name=name,
            cost_bps=COST_BPS,
        )

    # ------------------------------------------------------------------
    # 11. SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Summary of Findings")
    print("=" * 60)

    for name, ls_returns in strategy_dict.items():
        perf = compute_performance(ls_returns)
        r2_name = name.replace(" L/S", "")
        r2 = oos_r_squared(models[r2_name])
        print(f"\n{name}:")
        print(f"  Annualized Return: {perf['annual_return']:.2%}")
        print(f"  Annualized Vol:    {perf['annual_vol']:.2%}")
        print(f"  Sharpe Ratio:      {perf['sharpe']:.3f}")
        print(f"  Sortino Ratio:     {perf['sortino']:.3f}")
        print(f"  Max Drawdown:      {perf['max_drawdown']:.2%}")
        print(f"  OOS R²:            {r2:.4f} ({r2 * 100:.2f}%)")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("  - cumulative_returns.png")
    print("  - feature_importance.png")
    print("  - feature_importance_gbt.png")
    for name in models:
        safe = name.lower().replace(" ", "_")
        print(f"  - decile_returns_{safe}.png")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
