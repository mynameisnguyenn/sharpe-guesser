"""
Evaluation: Compare Strategies, Plot Results, Analyze Features
================================================================

This module provides the tools to assess whether your ML models actually
work. The Kelly paper evaluates models on two dimensions:
    1. Statistical: out-of-sample R-squared (spoiler: it's small, ~0.5%)
    2. Economic: can the predictions build a profitable long-short portfolio?

We focus on the economic evaluation because it's more intuitive and directly
answers the question "would you put money behind this model?"
"""

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .portfolio import compute_performance


# ---------------------------------------------------------------------------
# 1. STRATEGY COMPARISON TABLE
# ---------------------------------------------------------------------------

def compare_strategies(
    strategy_dict: dict[str, pd.Series],
    rf: float = 0.05,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.

    Parameters:
        strategy_dict:   {strategy_name: returns_series}
        rf:              Annual risk-free rate
        periods_per_year: 12 for monthly, 252 for daily

    Returns:
        DataFrame with one row per strategy, columns = performance metrics.
    """
    rows = {}
    for name, returns in strategy_dict.items():
        metrics = compute_performance(returns, rf=rf, periods_per_year=periods_per_year)
        rows[name] = metrics

    comparison = pd.DataFrame(rows).T
    comparison.index.name = "strategy"

    # Format for display
    print("\n" + "=" * 70)
    print("  Strategy Comparison")
    print("=" * 70)
    for col in comparison.columns:
        if col == "max_drawdown":
            comparison[col] = comparison[col].map(lambda x: f"{x:.1%}")
        else:
            comparison[col] = comparison[col].map(lambda x: f"{x:.3f}")
    print(comparison.to_string())
    print("=" * 70 + "\n")

    return comparison


# ---------------------------------------------------------------------------
# 2. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
# The Kelly paper's variable importance analysis is one of its most cited
# results. They find that momentum, short-term reversal, and liquidity
# dominate. Our random forest provides feature importances out of the box.

def feature_importance_plot(
    model: Any,
    feature_names: list[str],
    top_n: int = 15,
    save_path: str | None = None,
) -> None:
    """
    Bar chart of feature importances from a tree-based model.

    Uses the model's `feature_importances_` attribute (available on
    RandomForestRegressor, GradientBoostingRegressor, etc.).
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(indices)),
        importances[indices][::-1],
        color="#2196F3",
        edgecolor="none",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel("Importance (impurity reduction)")
    ax.set_title("Feature Importance — Random Forest")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. CUMULATIVE RETURN PLOT
# ---------------------------------------------------------------------------

def plot_cumulative(
    strategy_dict: dict[str, pd.Series],
    title: str = "Cumulative Returns",
    save_path: str | None = None,
) -> None:
    """
    Plot cumulative returns for multiple strategies on the same chart.

    This is the standard way to visualize strategy performance. The Kelly paper
    includes similar plots comparing their ML long-short portfolios to
    Fama-French benchmarks.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for i, (name, returns) in enumerate(strategy_dict.items()):
        cumulative = (1 + returns).cumprod()
        color = colors[i % len(colors)]
        ax.plot(cumulative.index, cumulative.values, label=name, color=color, linewidth=1.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return ($1 invested)")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. LONG-SHORT SPREAD VISUALIZATION
# ---------------------------------------------------------------------------

def plot_long_short_spread(
    long_returns: pd.Series,
    short_returns: pd.Series,
    title: str = "Long-Short Portfolio Decomposition",
    save_path: str | None = None,
) -> None:
    """
    Visualize the long leg, short leg, and spread of a long-short portfolio.

    This helps diagnose whether the strategy works because the long side
    picks winners, the short side picks losers, or both. In practice, most
    of the alpha usually comes from the short side (avoiding losers).
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: cumulative returns of each leg
    cum_long = (1 + long_returns).cumprod()
    cum_short = (1 + short_returns).cumprod()
    spread = long_returns - short_returns
    cum_spread = (1 + spread).cumprod()

    axes[0].plot(cum_long.index, cum_long.values, label="Long (top decile)",
                 color="#4CAF50", linewidth=1.5)
    axes[0].plot(cum_short.index, cum_short.values, label="Short (bottom decile)",
                 color="#FF5722", linewidth=1.5)
    axes[0].plot(cum_spread.index, cum_spread.values, label="Spread (L-S)",
                 color="#2196F3", linewidth=2)
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend(frameon=False)
    axes[0].set_title(title)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: monthly spread returns
    axes[1].bar(spread.index, spread.values, color="#2196F3", alpha=0.7, width=20)
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Monthly Spread Return")
    axes[1].set_xlabel("Date")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. DECILE RETURN BAR CHART
# ---------------------------------------------------------------------------

def plot_decile_returns(
    quantile_summary: pd.DataFrame,
    title: str = "Average Returns by Prediction Decile",
    save_path: str | None = None,
) -> None:
    """
    Bar chart of average returns by decile.

    A good model shows a monotonically increasing pattern from decile 1
    (lowest predicted) to decile 10 (highest predicted). If the pattern
    is flat or non-monotonic, the model isn't capturing real signal.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    deciles = quantile_summary.index
    returns = quantile_summary["annual_return"]

    colors = ["#FF5722" if r < 0 else "#4CAF50" for r in returns]
    ax.bar(deciles, returns, color=colors, edgecolor="none", alpha=0.85)

    ax.set_xlabel("Prediction Decile")
    ax.set_ylabel("Annualized Return")
    ax.set_title(title)
    ax.set_xticks(deciles)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
