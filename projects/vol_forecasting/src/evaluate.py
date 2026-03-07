"""
Evaluation: Loss Functions, Mincer-Zarnowitz Regression, Comparison Charts
==========================================================================

This module answers the core question: which vol forecast is best?

Volatility forecasting has its own set of loss functions — you can't just use
MSE because vol is right-skewed (big spikes matter more). The standard loss
in the academic literature is QLIKE, which penalizes under-prediction of high
vol more heavily than over-prediction. That's exactly what a risk manager wants:
it's worse to UNDERESTIMATE vol than to overestimate it.

The Mincer-Zarnowitz regression is the standard unbiasedness test:
    RV_actual = alpha + beta * forecast + epsilon
If alpha=0 and beta=1, the forecast is unbiased. R-squared tells you how
much of the variation in realized vol your forecast captures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# 1. LOSS FUNCTIONS
# ---------------------------------------------------------------------------

def qlike(realized: pd.Series, forecast: pd.Series) -> float:
    """
    QLIKE loss — the standard vol forecast evaluation metric.

    QLIKE = mean( realized/forecast - log(realized/forecast) - 1 )

    Lower is better. This is a quasi-likelihood loss derived from assuming
    returns are conditionally normal. It heavily penalizes under-prediction
    of high vol periods (like 2008, March 2020).
    """
    # Align and drop NaN/zero/negative values
    df = pd.DataFrame({"r": realized, "f": forecast}).dropna()
    df = df[(df["r"] > 0) & (df["f"] > 0)]
    if len(df) == 0:
        return np.nan
    ratio = df["r"] / df["f"]
    return (ratio - np.log(ratio) - 1).mean()


def mse(realized: pd.Series, forecast: pd.Series) -> float:
    """Mean squared error between realized and forecast vol."""
    df = pd.DataFrame({"r": realized, "f": forecast}).dropna()
    return ((df["r"] - df["f"]) ** 2).mean()


def mae(realized: pd.Series, forecast: pd.Series) -> float:
    """Mean absolute error between realized and forecast vol."""
    df = pd.DataFrame({"r": realized, "f": forecast}).dropna()
    return (df["r"] - df["f"]).abs().mean()


# ---------------------------------------------------------------------------
# 2. MINCER-ZARNOWITZ REGRESSION
# ---------------------------------------------------------------------------

def mincer_zarnowitz(realized: pd.Series, forecast: pd.Series, name: str = "Model") -> dict:
    """
    Mincer-Zarnowitz unbiasedness test.

    Regresses realized vol on forecast vol:
        RV = alpha + beta * forecast + epsilon

    A good forecast has alpha ≈ 0, beta ≈ 1, high R².

    Returns dict with regression results and prints a summary.
    """
    df = pd.DataFrame({"r": realized, "f": forecast}).dropna()
    if len(df) < 30:
        print(f"  {name}: insufficient data for MZ regression ({len(df)} obs)")
        return {}

    X = sm.add_constant(df["f"].values)
    result = sm.OLS(df["r"].values, X).fit()

    alpha = result.params[0]
    beta = result.params[1]
    r2 = result.rsquared

    print(f"\n  {name} — Mincer-Zarnowitz Regression:")
    print(f"    alpha:  {alpha:.4f} (t={result.tvalues[0]:.2f}, p={result.pvalues[0]:.3f})")
    print(f"    beta:   {beta:.4f} (t={result.tvalues[1]:.2f}, p={result.pvalues[1]:.3f})")
    print(f"    R²:     {r2:.4f} ({r2 * 100:.1f}%)")
    print(f"    N obs:  {len(df)}")

    return {
        "alpha": alpha,
        "beta": beta,
        "alpha_tstat": result.tvalues[0],
        "beta_tstat": result.tvalues[1],
        "r_squared": r2,
        "n_obs": len(df),
    }


# ---------------------------------------------------------------------------
# 3. MODEL COMPARISON TABLE
# ---------------------------------------------------------------------------

def compare_models(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Compare all forecast models using QLIKE, MSE, MAE, and MZ R².

    Parameters:
        forecasts: DataFrame with 'realized' column and model forecast columns.

    Returns:
        Summary DataFrame with one row per model.
    """
    realized = forecasts["realized"]
    model_cols = [c for c in forecasts.columns if c != "realized"]

    rows = {}
    for model in model_cols:
        forecast = forecasts[model]
        mz = mincer_zarnowitz(realized, forecast, name=model)
        rows[model] = {
            "QLIKE": qlike(realized, forecast),
            "MSE": mse(realized, forecast),
            "MAE": mae(realized, forecast),
            "MZ_R2": mz.get("r_squared", np.nan),
            "MZ_alpha": mz.get("alpha", np.nan),
            "MZ_beta": mz.get("beta", np.nan),
        }

    comparison = pd.DataFrame(rows).T
    comparison.index.name = "model"

    print("\n" + "=" * 70)
    print("  Vol Forecast Model Comparison")
    print("=" * 70)
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 70 + "\n")

    return comparison


# ---------------------------------------------------------------------------
# 4. TIME SERIES CHART — FORECAST vs REALIZED
# ---------------------------------------------------------------------------

def plot_forecast_vs_realized(
    forecasts: pd.DataFrame,
    title: str = "Volatility Forecasts vs Realized",
    save_path: str | None = None,
) -> None:
    """
    Time series plot of all forecasts overlaid with realized vol.

    The realized vol line is thick and dark — the ground truth.
    Forecast lines are thinner and colored.
    """
    realized = forecasts["realized"]
    model_cols = [c for c in forecasts.columns if c != "realized"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Realized vol — ground truth
    ax.plot(realized.index, realized.values, color="black", linewidth=2,
            label="Realized", alpha=0.7)

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    for i, model in enumerate(model_cols):
        ax.plot(forecasts[model].index, forecasts[model].values,
                color=colors[i % len(colors)], linewidth=1.2,
                label=model.upper(), alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. MZ SCATTER PLOTS
# ---------------------------------------------------------------------------

def plot_mz_scatter(
    forecasts: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """
    Scatter plots of realized vs forecast vol for each model.

    Each subplot shows one model with the 45-degree line (perfect forecast)
    and the OLS fit line.
    """
    realized = forecasts["realized"]
    model_cols = [c for c in forecasts.columns if c != "realized"]
    n_models = len(model_cols)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for i, model in enumerate(model_cols):
        ax = axes[i]
        df = pd.DataFrame({"r": realized, "f": forecasts[model]}).dropna()

        ax.scatter(df["f"], df["r"], alpha=0.15, s=8, color=colors[i % len(colors)])

        # 45-degree line
        lims = [min(df["f"].min(), df["r"].min()), max(df["f"].max(), df["r"].max())]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="Perfect")

        # OLS fit line
        X = sm.add_constant(df["f"].values)
        result = sm.OLS(df["r"].values, X).fit()
        x_fit = np.linspace(lims[0], lims[1], 100)
        y_fit = result.params[0] + result.params[1] * x_fit
        ax.plot(x_fit, y_fit, color=colors[i % len(colors)], linewidth=1.5,
                label=f"OLS (R²={result.rsquared:.2f})")

        ax.set_xlabel("Forecast Vol")
        ax.set_ylabel("Realized Vol")
        ax.set_title(model.upper())
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Mincer-Zarnowitz: Realized vs Forecast", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. VOL CONE CHART
# ---------------------------------------------------------------------------

def plot_vol_cone(
    vol_cone_df: pd.DataFrame,
    current_rv: dict | None = None,
    title: str = "Volatility Cone",
    save_path: str | None = None,
) -> None:
    """
    Plot the vol cone — percentiles of realized vol at different horizons.

    Optionally overlay current realized vol at each horizon.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    windows = vol_cone_df.index.values

    ax.fill_between(windows, vol_cone_df["10th"], vol_cone_df["90th"],
                    alpha=0.15, color="#2196F3", label="10th-90th")
    ax.fill_between(windows, vol_cone_df["25th"], vol_cone_df["75th"],
                    alpha=0.3, color="#2196F3", label="25th-75th")
    ax.plot(windows, vol_cone_df["50th"], color="#2196F3", linewidth=2,
            label="Median")

    if current_rv is not None:
        curr_windows = sorted(current_rv.keys())
        curr_vals = [current_rv[w] for w in curr_windows]
        ax.plot(curr_windows, curr_vals, "ro-", markersize=8, linewidth=2,
                label="Current", zorder=5)

    ax.set_xlabel("Window (trading days)")
    ax.set_ylabel("Annualized Volatility")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. VOL TARGETING COMPARISON CHART
# ---------------------------------------------------------------------------

def plot_vol_target_comparison(
    results: dict,
    save_path: str | None = None,
) -> None:
    """
    Compare unmanaged vs vol-targeted portfolio.

    Parameters:
        results: dict with keys 'unmanaged' and 'managed', each containing
                 a dict with 'cumulative', 'rolling_vol', and 'metrics'.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: cumulative returns
    for label, data in results.items():
        cum = data["cumulative"]
        lw = 2 if label == "managed" else 1.5
        axes[0].plot(cum.index, cum.values, label=label.title(), linewidth=lw)

    axes[0].set_ylabel("Cumulative Return ($1)")
    axes[0].set_title("Vol-Targeted vs Unmanaged Portfolio")
    axes[0].legend(frameon=False)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(True, alpha=0.3)

    # Bottom: rolling vol
    for label, data in results.items():
        rv = data["rolling_vol"]
        axes[1].plot(rv.index, rv.values, label=label.title(), linewidth=1.2)

    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Rolling 63-day Vol (annualized)")
    axes[1].legend(frameon=False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
