"""
Risk Metrics: Regime-Conditional VaR, CVaR, and Volatility
===========================================================

The central argument: unconditional risk metrics (standard VaR, CVaR, vol)
blend calm and crisis periods, understating tail risk. Regime-conditional
metrics separate these environments and reveal the true risk in each state.

A PM who only sees unconditional 95% VaR of -1.5% doesn't realize that
crisis-regime VaR might be -3.5%. This module quantifies that gap.
"""

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def conditional_var(
    returns: pd.Series,
    regimes: pd.Series,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Compute VaR (historical percentile) per regime and unconditional.

    Returns DataFrame with regime labels as index, VaR as column.
    """
    aligned = pd.DataFrame({"ret": returns, "regime": regimes}).dropna()
    rows = []

    # Unconditional
    rows.append({
        "regime": "Unconditional",
        "VaR": aligned["ret"].quantile(1 - confidence),
        "days": len(aligned),
    })

    for regime in sorted(aligned["regime"].unique()):
        sub = aligned.loc[aligned["regime"] == regime, "ret"]
        rows.append({
            "regime": regime,
            "VaR": sub.quantile(1 - confidence),
            "days": len(sub),
        })

    return pd.DataFrame(rows).set_index("regime")


def conditional_cvar(
    returns: pd.Series,
    regimes: pd.Series,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Compute CVaR (Expected Shortfall) per regime and unconditional.

    CVaR = average return on days worse than VaR. Always worse than VaR.
    """
    aligned = pd.DataFrame({"ret": returns, "regime": regimes}).dropna()
    rows = []

    # Unconditional
    var_unc = aligned["ret"].quantile(1 - confidence)
    tail_unc = aligned.loc[aligned["ret"] <= var_unc, "ret"]
    rows.append({
        "regime": "Unconditional",
        "CVaR": tail_unc.mean() if len(tail_unc) > 0 else var_unc,
        "days": len(aligned),
    })

    for regime in sorted(aligned["regime"].unique()):
        sub = aligned.loc[aligned["regime"] == regime, "ret"]
        var_r = sub.quantile(1 - confidence)
        tail = sub[sub <= var_r]
        rows.append({
            "regime": regime,
            "CVaR": tail.mean() if len(tail) > 0 else var_r,
            "days": len(sub),
        })

    return pd.DataFrame(rows).set_index("regime")


def conditional_vol(
    returns: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Annualized volatility per regime and unconditional."""
    aligned = pd.DataFrame({"ret": returns, "regime": regimes}).dropna()
    rows = []

    rows.append({
        "regime": "Unconditional",
        "vol": aligned["ret"].std() * np.sqrt(TRADING_DAYS),
        "days": len(aligned),
    })

    for regime in sorted(aligned["regime"].unique()):
        sub = aligned.loc[aligned["regime"] == regime, "ret"]
        rows.append({
            "regime": regime,
            "vol": sub.std() * np.sqrt(TRADING_DAYS),
            "days": len(sub),
        })

    return pd.DataFrame(rows).set_index("regime")


def regime_summary_table(
    returns: pd.Series,
    prices: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    Comprehensive summary per regime: days, proportion, return, vol,
    Sharpe, VaR (95%), max drawdown.
    """
    aligned = pd.DataFrame({
        "ret": returns,
        "price": prices,
        "regime": regimes,
    }).dropna()

    total_days = len(aligned)
    rows = []

    for regime in sorted(aligned["regime"].unique()):
        mask = aligned["regime"] == regime
        sub = aligned.loc[mask]
        rets = sub["ret"]
        n = len(rets)

        ann_return = rets.mean() * TRADING_DAYS
        ann_vol = rets.std() * np.sqrt(TRADING_DAYS)
        sharpe = ann_return / ann_vol if ann_vol > 1e-10 else 0.0
        var_95 = rets.quantile(0.05)

        # Max drawdown within regime periods
        cum = (1 + rets).cumprod()
        cummax = cum.cummax()
        dd = (cum - cummax) / cummax
        max_dd = dd.min()

        rows.append({
            "regime": regime,
            "days": n,
            "proportion": n / total_days,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "var_95": var_95,
            "max_drawdown": max_dd,
        })

    return pd.DataFrame(rows).set_index("regime")
