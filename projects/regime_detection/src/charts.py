"""
Charts: Static Matplotlib Charts for Results Directory
=======================================================

Four publication-quality charts:
1. Regime timeline — SPY price with regime-colored background (hero chart)
2. Factor correlation heatmaps — one per regime showing correlation breakdown
3. Return distributions — overlaid histograms by regime
4. Transition matrix — annotated heatmap of transition probabilities

All charts follow the project convention: optional save_path, plt.close(fig),
no plt.show().
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


REGIME_COLORS = {
    "Bull": "#2ecc71",
    "Stress": "#f39c12",
    "Crisis": "#e74c3c",
}


def plot_regime_timeline(
    prices: pd.Series,
    regimes: pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """SPY price chart with regime-colored background shading."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(prices.index, prices.values, color="black", linewidth=0.8, label="SPY")

    # Color background by regime
    valid = regimes.dropna()
    if len(valid) > 0:
        prev_regime = valid.iloc[0]
        start_date = valid.index[0]

        for i in range(1, len(valid)):
            curr_regime = valid.iloc[i]
            if curr_regime != prev_regime or i == len(valid) - 1:
                end_date = valid.index[i]
                color = REGIME_COLORS.get(prev_regime, "#cccccc")
                ax.axvspan(start_date, end_date, alpha=0.25, color=color)
                prev_regime = curr_regime
                start_date = end_date

    # Legend patches for regimes
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=REGIME_COLORS.get(r, "#ccc"), alpha=0.4, label=r)
        for r in sorted(REGIME_COLORS.keys())
        if r in regimes.dropna().unique()
    ]
    legend_patches.insert(0, plt.Line2D([0], [0], color="black", linewidth=0.8, label="SPY"))
    ax.legend(handles=legend_patches, loc="upper left", framealpha=0.9)

    ax.set_title("SPY Price with HMM-Detected Market Regimes", fontsize=13, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_factor_heatmaps(
    correlations_by_regime: dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """1xN grid of correlation matrices, one per regime."""
    n = len(correlations_by_regime)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (regime, corr) in zip(axes, sorted(correlations_by_regime.items())):
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, center=0, ax=ax,
            square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"{regime} Regime", fontsize=11, fontweight="bold")

    fig.suptitle("Factor Correlations by Market Regime", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_regime_distributions(
    returns: pd.Series,
    regimes: pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlaid return histograms by regime."""
    fig, ax = plt.subplots(figsize=(10, 5))

    aligned = pd.DataFrame({"ret": returns, "regime": regimes}).dropna()

    for regime in sorted(aligned["regime"].unique()):
        sub = aligned.loc[aligned["regime"] == regime, "ret"]
        color = REGIME_COLORS.get(regime, "#999999")
        ax.hist(
            sub, bins=80, alpha=0.5, color=color, label=regime,
            density=True, edgecolor="none",
        )

    ax.set_title("Daily Return Distributions by Regime", fontsize=13, fontweight="bold")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_transition_matrix(
    trans_matrix: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Annotated heatmap of transition probabilities."""
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        trans_matrix, annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0, vmax=1, ax=ax, square=True,
        linewidths=1, linecolor="white",
        cbar_kws={"label": "Probability"},
    )

    ax.set_title("Regime Transition Probabilities", fontsize=13, fontweight="bold")
    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
