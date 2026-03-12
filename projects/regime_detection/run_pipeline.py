"""
Regime Detection Pipeline — End-to-End Orchestration
=====================================================

Runs the full regime detection analysis:
    1. Fetch SPY, VIX, and FF daily factors
    2. Build HMM features (returns, VIX z-score, realized vol)
    3. Fit HMM on full sample (for transition matrix and summary stats)
    4. Walk-forward regime detection (out-of-sample regime labels)
    5. Factor analysis by regime (stats, correlations, exposures)
    6. Regime-conditional risk metrics (VaR, CVaR, vol)
    7. Generate static charts → results/
    8. Print summary

Run:
    cd projects/regime_detection
    python run_pipeline.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pandas as pd

from data.fetch_data import fetch_spy_vix, fetch_ff_daily
from src.regime_model import (
    build_hmm_features, fit_hmm, label_regimes, walk_forward_regimes,
    transition_matrix, expected_duration,
)
from src.factor_analysis import (
    factor_stats_by_regime, correlation_by_regime, regime_factor_exposures,
)
from src.risk_metrics import (
    conditional_var, conditional_cvar, conditional_vol, regime_summary_table,
)
from src.charts import (
    plot_regime_timeline, plot_factor_heatmaps,
    plot_regime_distributions, plot_transition_matrix,
)


RESULTS_DIR = Path(__file__).parent / "results"
N_STATES = 3


def main():
    print("=" * 60)
    print("  Regime Detection — Full Pipeline")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Step 1: Fetch data
    # ------------------------------------------------------------------
    print("Step 1/8: Fetching data...")
    spy_prices, vix_df = fetch_spy_vix()
    ff_daily = fetch_ff_daily()

    spy_returns = spy_prices["SPY"].pct_change().dropna()
    spy_returns.name = "returns"
    vix = vix_df["VIX"]

    # Align FF factors to SPY dates
    factor_cols = ["Mkt-RF", "SMB", "HML"]
    ff_aligned = ff_daily[factor_cols].reindex(spy_returns.index).dropna()
    print(f"  SPY: {len(spy_returns)} days, FF factors: {len(ff_aligned)} days")
    print()

    # ------------------------------------------------------------------
    # Step 2: Build HMM features
    # ------------------------------------------------------------------
    print("Step 2/8: Building HMM features...")
    features = build_hmm_features(spy_returns, vix)
    print(f"  Features: {len(features)} days x {len(features.columns)} columns")
    print()

    # ------------------------------------------------------------------
    # Step 3: Fit HMM on full sample
    # ------------------------------------------------------------------
    print(f"Step 3/8: Fitting {N_STATES}-state HMM on full sample...")
    model = fit_hmm(features, n_states=N_STATES)
    labels = label_regimes(model)
    trans = transition_matrix(model)
    durations = expected_duration(model)

    print("  Regime labels:", labels)
    print("  Expected durations (trading days):")
    for name, dur in sorted(durations.items()):
        print(f"    {name}: {dur:.0f} days ({dur / 252:.1f} years)")
    print()

    # ------------------------------------------------------------------
    # Step 4: Walk-forward regime detection
    # ------------------------------------------------------------------
    print("Step 4/8: Walk-forward regime detection...")
    regimes = walk_forward_regimes(
        features, n_states=N_STATES,
        min_history=504, refit_every=63,
    )
    valid_regimes = regimes.dropna()
    regime_counts = valid_regimes.value_counts()
    print(f"  Classified {len(valid_regimes)} days:")
    for regime, count in regime_counts.items():
        pct = count / len(valid_regimes) * 100
        print(f"    {regime}: {count} days ({pct:.1f}%)")
    print()

    # ------------------------------------------------------------------
    # Step 5: Factor analysis by regime
    # ------------------------------------------------------------------
    print("Step 5/8: Factor analysis by regime...")
    factor_stats = factor_stats_by_regime(ff_aligned, regimes)
    corr_by_regime = correlation_by_regime(ff_aligned, regimes)
    exposures = regime_factor_exposures(spy_returns, ff_aligned, regimes)

    print("  Factor Sharpe ratios by regime:")
    for regime in factor_stats.index:
        sharpes = {
            col.replace("_sharpe", ""): f"{factor_stats.loc[regime, col]:.2f}"
            for col in factor_stats.columns if col.endswith("_sharpe")
        }
        print(f"    {regime}: {sharpes}")
    print()

    # ------------------------------------------------------------------
    # Step 6: Risk metrics
    # ------------------------------------------------------------------
    print("Step 6/8: Computing regime-conditional risk metrics...")
    var_df = conditional_var(spy_returns, regimes)
    cvar_df = conditional_cvar(spy_returns, regimes)
    vol_df = conditional_vol(spy_returns, regimes)

    # Align prices to regime dates
    spy_prices_aligned = spy_prices["SPY"].reindex(spy_returns.index).ffill()
    summary = regime_summary_table(spy_returns, spy_prices_aligned, regimes)

    print("\n  Regime Summary:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n  VaR (95%) by regime:")
    for regime in var_df.index:
        print(f"    {regime}: {var_df.loc[regime, 'VaR']:.4f}")

    print("\n  CVaR (95%) by regime:")
    for regime in cvar_df.index:
        print(f"    {regime}: {cvar_df.loc[regime, 'CVaR']:.4f}")
    print()

    # ------------------------------------------------------------------
    # Step 7: Generate charts
    # ------------------------------------------------------------------
    print("Step 7/8: Generating charts...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_regime_timeline(
        spy_prices["SPY"], regimes,
        save_path=RESULTS_DIR / "regime_timeline.png",
    )
    print("  Saved: regime_timeline.png")

    plot_factor_heatmaps(
        corr_by_regime,
        save_path=RESULTS_DIR / "factor_correlations.png",
    )
    print("  Saved: factor_correlations.png")

    plot_regime_distributions(
        spy_returns, regimes,
        save_path=RESULTS_DIR / "regime_distributions.png",
    )
    print("  Saved: regime_distributions.png")

    plot_transition_matrix(
        trans,
        save_path=RESULTS_DIR / "transition_matrix.png",
    )
    print("  Saved: transition_matrix.png")
    print()

    # ------------------------------------------------------------------
    # Step 8: Summary
    # ------------------------------------------------------------------
    print("Step 8/8: Pipeline complete!")
    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print()
    print(f"Transition matrix:")
    print(trans.to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"Risk underestimation:")
    if "Crisis" in var_df.index and "Unconditional" in var_df.index:
        unc_var = var_df.loc["Unconditional", "VaR"]
        crisis_var = var_df.loc["Crisis", "VaR"]
        ratio = crisis_var / unc_var
        print(f"  Unconditional 95% VaR: {unc_var:.4f}")
        print(f"  Crisis 95% VaR:        {crisis_var:.4f}")
        print(f"  Crisis VaR is {ratio:.1f}x worse than unconditional")
    print()
    print(f"Charts saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
