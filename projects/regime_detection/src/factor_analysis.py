"""
Factor Analysis: How Factor Exposures and Correlations Shift by Regime
=======================================================================

The key insight: unconditional factor statistics blend very different
market environments. In a crisis, correlations spike toward 1 (diversification
fails exactly when you need it), factor premiums collapse, and exposures shift.

This module separates factor behavior by regime so a PM or CRO can see:
- Which factors work in each regime
- How correlation structure breaks down in stress
- How beta/exposure to each factor changes across regimes
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


TRADING_DAYS = 252


def factor_stats_by_regime(
    factor_returns: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    Compute annualized mean, vol, Sharpe, and skew per factor per regime.

    Returns a MultiIndex DataFrame: (regime, stat) x factors.
    """
    aligned = factor_returns.join(regimes.rename("regime"), how="inner").dropna(subset=["regime"])
    results = []

    for regime in sorted(aligned["regime"].unique()):
        mask = aligned["regime"] == regime
        sub = aligned.loc[mask, factor_returns.columns]
        n_days = len(sub)

        row = {"regime": regime, "days": n_days}
        for col in factor_returns.columns:
            s = sub[col]
            ann_mean = s.mean() * TRADING_DAYS
            ann_vol = s.std() * np.sqrt(TRADING_DAYS)
            sharpe = ann_mean / ann_vol if ann_vol > 1e-10 else 0.0
            row[f"{col}_mean"] = ann_mean
            row[f"{col}_vol"] = ann_vol
            row[f"{col}_sharpe"] = sharpe
            row[f"{col}_skew"] = s.skew()

        results.append(row)

    return pd.DataFrame(results).set_index("regime")


def correlation_by_regime(
    factor_returns: pd.DataFrame,
    regimes: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Return a dict of correlation matrices, one per regime."""
    aligned = factor_returns.join(regimes.rename("regime"), how="inner").dropna(subset=["regime"])
    correlations = {}

    for regime in sorted(aligned["regime"].unique()):
        mask = aligned["regime"] == regime
        sub = aligned.loc[mask, factor_returns.columns]
        correlations[regime] = sub.corr()

    return correlations


def regime_factor_exposures(
    spy_returns: pd.Series,
    factors: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    OLS regression of SPY excess returns on factors, separately per regime.

    Returns DataFrame with alpha, betas, R-squared per regime.
    """
    aligned = pd.DataFrame({"spy": spy_returns}).join(factors).join(
        regimes.rename("regime")
    ).dropna()

    results = []
    factor_cols = factors.columns.tolist()

    for regime in sorted(aligned["regime"].unique()):
        mask = aligned["regime"] == regime
        sub = aligned.loc[mask]

        y = sub["spy"]
        X = sm.add_constant(sub[factor_cols])

        try:
            model = sm.OLS(y, X).fit()
            row = {
                "regime": regime,
                "alpha_annual": model.params.get("const", 0.0) * TRADING_DAYS,
                "r_squared": model.rsquared,
                "n_days": len(sub),
            }
            for fc in factor_cols:
                row[f"beta_{fc}"] = model.params.get(fc, 0.0)
                row[f"tstat_{fc}"] = model.tvalues.get(fc, 0.0)
            results.append(row)
        except Exception:
            continue

    return pd.DataFrame(results).set_index("regime")
