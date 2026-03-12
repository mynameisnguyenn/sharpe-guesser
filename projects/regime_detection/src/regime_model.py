"""
Regime Model: Hidden Markov Model for Market Regime Detection
==============================================================

Fits a Gaussian HMM to market features (returns, VIX z-score, realized vol)
to classify trading days into 2-3 regimes (Bull / Stress / Crisis).

Key design decisions:
- VIX is z-scored with an expanding window to avoid lookahead bias
- Walk-forward regime detection with expanding window + periodic refit
- States are sorted by mean return after each fit to handle label switching
- Convergence failures carry forward the last fitted model (same pattern as GARCH)
"""

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


TRADING_DAYS = 252


def build_hmm_features(
    spy_returns: pd.Series,
    vix: pd.Series,
    rv_window: int = 22,
) -> pd.DataFrame:
    """
    Build feature matrix for HMM: [return, vix_z, rv].

    VIX is z-scored with an expanding window (min 63 days) to avoid lookahead.
    Realized vol is 22-day rolling std, annualized.
    """
    # Align on common dates
    combined = pd.DataFrame({
        "return": spy_returns,
        "vix": vix,
    }).dropna()

    # Z-score VIX with expanding window (no lookahead)
    vix_mean = combined["vix"].expanding(min_periods=63).mean()
    vix_std = combined["vix"].expanding(min_periods=63).std()
    combined["vix_z"] = (combined["vix"] - vix_mean) / vix_std

    # Realized vol: annualized rolling std of returns
    combined["rv"] = combined["return"].rolling(rv_window).std() * np.sqrt(TRADING_DAYS)

    # Keep only the 3 HMM features
    features = combined[["return", "vix_z", "rv"]].dropna()
    return features


def fit_hmm(features: pd.DataFrame, n_states: int = 3) -> GaussianHMM:
    """Fit a Gaussian HMM to the feature matrix."""
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(features.values)
    return model


def label_regimes(model: GaussianHMM) -> dict[int, str]:
    """
    Sort states by mean return and assign labels.

    Lowest mean return → Crisis, highest → Bull.
    For 2-state: lowest → Stress, highest → Bull.
    """
    n = model.n_components
    # Mean return is the first feature (column 0)
    mean_returns = model.means_[:, 0]
    sorted_indices = np.argsort(mean_returns)

    if n == 2:
        labels = {int(sorted_indices[0]): "Stress", int(sorted_indices[1]): "Bull"}
    else:
        names = ["Crisis", "Stress", "Bull"]
        labels = {int(sorted_indices[i]): names[i] for i in range(n)}

    return labels


def walk_forward_regimes(
    features: pd.DataFrame,
    n_states: int = 3,
    min_history: int = 504,
    refit_every: int = 63,
) -> pd.Series:
    """
    Walk-forward regime detection with expanding window.

    Refits the HMM every `refit_every` days using all data up to that point.
    Between refits, uses the last fitted model to predict the current regime.
    States are re-sorted by mean return after each fit to handle label switching.
    """
    n = len(features)
    # Store string labels directly (not numeric states) to avoid label switching
    regimes = pd.Series(dtype="object", index=features.index, name="regime")

    if n < min_history:
        return regimes

    last_model = None
    last_fit_idx = None
    last_labels = None

    for t in range(min_history, n):
        need_refit = (
            last_model is None
            or (t - last_fit_idx) >= refit_every
        )

        if need_refit:
            train = features.iloc[:t]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = fit_hmm(train, n_states=n_states)
                    last_model = model
                    last_labels = label_regimes(model)
                    last_fit_idx = t
                except Exception:
                    # Convergence failure — carry forward last model
                    if last_model is None:
                        continue

        if last_model is None:
            continue

        # Predict regime and map to label immediately (handles label switching between refits)
        obs = features.iloc[t:t + 1].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                state = last_model.predict(obs)[0]
                regimes.iloc[t] = last_labels[state]
            except Exception:
                continue

    regimes.name = "regime"
    return regimes


def transition_matrix(model: GaussianHMM) -> pd.DataFrame:
    """Return the transition matrix as a labeled DataFrame."""
    labels = label_regimes(model)
    n = model.n_components
    sorted_indices = sorted(labels.keys(), key=lambda x: labels[x])
    sorted_names = [labels[i] for i in sorted_indices]

    # Reorder transition matrix rows and columns by label order
    trans = model.transmat_[np.ix_(sorted_indices, sorted_indices)]
    return pd.DataFrame(trans, index=sorted_names, columns=sorted_names)


def expected_duration(model: GaussianHMM) -> dict[str, float]:
    """Expected duration in each regime: 1 / (1 - p_ii) trading days."""
    labels = label_regimes(model)
    durations = {}
    for state_idx, name in labels.items():
        p_stay = model.transmat_[state_idx, state_idx]
        if p_stay < 1.0:
            durations[name] = 1.0 / (1.0 - p_stay)
        else:
            durations[name] = np.inf
    return durations
