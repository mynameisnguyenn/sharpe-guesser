"""
Tests for regime detection project.

All tests use synthetic data — no network calls, runs instantly.
Covers: regime_model, factor_analysis, risk_metrics, charts.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures — synthetic data with built-in regime structure
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_dates():
    """Three years of trading days (756)."""
    return pd.bdate_range("2020-01-01", periods=756, freq="B")


@pytest.fixture
def daily_returns(daily_dates):
    """Synthetic daily returns with 3 distinct regimes."""
    np.random.seed(42)
    n = len(daily_dates)
    third = n // 3
    # Bull: positive drift, low vol
    bull = np.random.normal(0.0005, 0.008, third)
    # Stress: near-zero drift, medium vol
    stress = np.random.normal(0.0, 0.015, third)
    # Crisis: negative drift, high vol
    crisis = np.random.normal(-0.002, 0.030, n - 2 * third)
    returns = np.concatenate([bull, stress, crisis])
    return pd.Series(returns, index=daily_dates, name="returns")


@pytest.fixture
def daily_prices(daily_returns):
    """Synthetic prices from cumulative returns."""
    return (100 * (1 + daily_returns).cumprod()).rename("SPY")


@pytest.fixture
def synthetic_vix(daily_dates):
    """Synthetic VIX matching regime structure."""
    np.random.seed(123)
    n = len(daily_dates)
    third = n // 3
    bull_vix = np.random.normal(14, 2, third).clip(10, 20)
    stress_vix = np.random.normal(22, 3, third).clip(15, 35)
    crisis_vix = np.random.normal(35, 8, n - 2 * third).clip(20, 80)
    vix = np.concatenate([bull_vix, stress_vix, crisis_vix])
    return pd.Series(vix, index=daily_dates, name="VIX")


@pytest.fixture
def deterministic_regimes(daily_dates):
    """Known regime labels for downstream tests."""
    n = len(daily_dates)
    third = n // 3
    labels = (["Bull"] * third + ["Stress"] * third + ["Crisis"] * (n - 2 * third))
    return pd.Series(labels, index=daily_dates, name="regime")


@pytest.fixture
def factor_returns(daily_dates):
    """Synthetic daily factor returns (Mkt-RF, SMB, HML)."""
    np.random.seed(77)
    n = len(daily_dates)
    return pd.DataFrame({
        "Mkt-RF": np.random.normal(0.0003, 0.012, n),
        "SMB": np.random.normal(0.0001, 0.006, n),
        "HML": np.random.normal(0.00005, 0.005, n),
    }, index=daily_dates)


@pytest.fixture
def hmm_features(daily_returns, synthetic_vix):
    """Build HMM features from synthetic data."""
    from src.regime_model import build_hmm_features
    return build_hmm_features(daily_returns, synthetic_vix)


# ---------------------------------------------------------------------------
# HMM Feature Tests
# ---------------------------------------------------------------------------

class TestBuildHMMFeatures:
    def test_output_columns(self, hmm_features):
        assert set(hmm_features.columns) == {"return", "vix_z", "rv"}

    def test_no_nans(self, hmm_features):
        assert hmm_features.isna().sum().sum() == 0

    def test_vix_z_finite(self, hmm_features):
        # Z-scored VIX should be finite (expanding window biases mean with regime shift)
        assert hmm_features["vix_z"].isna().sum() == 0
        assert np.isfinite(hmm_features["vix_z"]).all()

    def test_rv_positive(self, hmm_features):
        assert (hmm_features["rv"] > 0).all()

    def test_length_less_than_input(self, daily_returns, hmm_features):
        # Some rows dropped due to expanding window + rolling
        assert len(hmm_features) < len(daily_returns)


# ---------------------------------------------------------------------------
# HMM Model Tests
# ---------------------------------------------------------------------------

class TestFitHMM:
    def test_returns_model(self, hmm_features):
        from src.regime_model import fit_hmm
        model = fit_hmm(hmm_features, n_states=3)
        assert model.n_components == 3

    def test_two_states(self, hmm_features):
        from src.regime_model import fit_hmm
        model = fit_hmm(hmm_features, n_states=2)
        assert model.n_components == 2

    def test_transition_rows_sum_to_one(self, hmm_features):
        from src.regime_model import fit_hmm
        model = fit_hmm(hmm_features, n_states=3)
        row_sums = model.transmat_.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)


class TestLabelRegimes:
    def test_unique_labels_3state(self, hmm_features):
        from src.regime_model import fit_hmm, label_regimes
        model = fit_hmm(hmm_features, n_states=3)
        labels = label_regimes(model)
        assert len(set(labels.values())) == 3
        assert set(labels.values()) == {"Bull", "Stress", "Crisis"}

    def test_unique_labels_2state(self, hmm_features):
        from src.regime_model import fit_hmm, label_regimes
        model = fit_hmm(hmm_features, n_states=2)
        labels = label_regimes(model)
        assert len(set(labels.values())) == 2
        assert set(labels.values()) == {"Bull", "Stress"}

    def test_bull_highest_mean_return(self, hmm_features):
        from src.regime_model import fit_hmm, label_regimes
        model = fit_hmm(hmm_features, n_states=3)
        labels = label_regimes(model)
        # Bull state should have highest mean return
        bull_idx = [k for k, v in labels.items() if v == "Bull"][0]
        crisis_idx = [k for k, v in labels.items() if v == "Crisis"][0]
        assert model.means_[bull_idx, 0] > model.means_[crisis_idx, 0]


class TestTransitionMatrix:
    def test_shape(self, hmm_features):
        from src.regime_model import fit_hmm, transition_matrix
        model = fit_hmm(hmm_features, n_states=3)
        tm = transition_matrix(model)
        assert tm.shape == (3, 3)

    def test_rows_sum_to_one(self, hmm_features):
        from src.regime_model import fit_hmm, transition_matrix
        model = fit_hmm(hmm_features, n_states=3)
        tm = transition_matrix(model)
        assert np.allclose(tm.sum(axis=1), 1.0, atol=1e-6)

    def test_labeled_index(self, hmm_features):
        from src.regime_model import fit_hmm, transition_matrix
        model = fit_hmm(hmm_features, n_states=3)
        tm = transition_matrix(model)
        assert all(isinstance(idx, str) for idx in tm.index)


class TestExpectedDuration:
    def test_positive_durations(self, hmm_features):
        from src.regime_model import fit_hmm, expected_duration
        model = fit_hmm(hmm_features, n_states=3)
        durations = expected_duration(model)
        assert all(d > 0 for d in durations.values())

    def test_at_least_one_day(self, hmm_features):
        from src.regime_model import fit_hmm, expected_duration
        model = fit_hmm(hmm_features, n_states=3)
        durations = expected_duration(model)
        assert all(d >= 1.0 for d in durations.values())


class TestWalkForwardRegimes:
    def test_output_length(self, hmm_features):
        from src.regime_model import walk_forward_regimes
        regimes = walk_forward_regimes(hmm_features, n_states=3, min_history=504, refit_every=63)
        assert len(regimes) == len(hmm_features)

    def test_has_valid_labels(self, hmm_features):
        from src.regime_model import walk_forward_regimes
        regimes = walk_forward_regimes(hmm_features, n_states=3, min_history=504, refit_every=63)
        valid = regimes.dropna()
        assert len(valid) > 0
        assert set(valid.unique()).issubset({"Bull", "Stress", "Crisis"})

    def test_insufficient_history(self, hmm_features):
        from src.regime_model import walk_forward_regimes
        short = hmm_features.iloc[:100]
        regimes = walk_forward_regimes(short, n_states=3, min_history=504)
        assert regimes.isna().all()


# ---------------------------------------------------------------------------
# Factor Analysis Tests
# ---------------------------------------------------------------------------

class TestFactorStatsByRegime:
    def test_output_has_all_regimes(self, factor_returns, deterministic_regimes):
        from src.factor_analysis import factor_stats_by_regime
        stats = factor_stats_by_regime(factor_returns, deterministic_regimes)
        assert "Bull" in stats.index
        assert "Stress" in stats.index
        assert "Crisis" in stats.index

    def test_days_sum_to_total(self, factor_returns, deterministic_regimes):
        from src.factor_analysis import factor_stats_by_regime
        stats = factor_stats_by_regime(factor_returns, deterministic_regimes)
        total = stats["days"].sum()
        assert total == len(factor_returns)


class TestCorrelationByRegime:
    def test_returns_dict(self, factor_returns, deterministic_regimes):
        from src.factor_analysis import correlation_by_regime
        corrs = correlation_by_regime(factor_returns, deterministic_regimes)
        assert isinstance(corrs, dict)
        assert len(corrs) == 3

    def test_diagonal_is_one(self, factor_returns, deterministic_regimes):
        from src.factor_analysis import correlation_by_regime
        corrs = correlation_by_regime(factor_returns, deterministic_regimes)
        for regime, corr_matrix in corrs.items():
            diag = np.diag(corr_matrix.values)
            assert np.allclose(diag, 1.0, atol=1e-10)

    def test_symmetric(self, factor_returns, deterministic_regimes):
        from src.factor_analysis import correlation_by_regime
        corrs = correlation_by_regime(factor_returns, deterministic_regimes)
        for regime, corr_matrix in corrs.items():
            assert np.allclose(corr_matrix.values, corr_matrix.values.T, atol=1e-10)


class TestRegimeFactorExposures:
    def test_output_has_betas(self, daily_returns, factor_returns, deterministic_regimes):
        from src.factor_analysis import regime_factor_exposures
        exposures = regime_factor_exposures(daily_returns, factor_returns, deterministic_regimes)
        assert "beta_Mkt-RF" in exposures.columns
        assert "r_squared" in exposures.columns

    def test_r_squared_bounded(self, daily_returns, factor_returns, deterministic_regimes):
        from src.factor_analysis import regime_factor_exposures
        exposures = regime_factor_exposures(daily_returns, factor_returns, deterministic_regimes)
        assert (exposures["r_squared"] >= 0).all()
        assert (exposures["r_squared"] <= 1).all()


# ---------------------------------------------------------------------------
# Risk Metrics Tests
# ---------------------------------------------------------------------------

class TestConditionalVaR:
    def test_has_unconditional(self, daily_returns, deterministic_regimes):
        from src.risk_metrics import conditional_var
        var_df = conditional_var(daily_returns, deterministic_regimes)
        assert "Unconditional" in var_df.index

    def test_var_is_negative(self, daily_returns, deterministic_regimes):
        from src.risk_metrics import conditional_var
        var_df = conditional_var(daily_returns, deterministic_regimes)
        assert (var_df["VaR"] < 0).all()

    def test_crisis_var_worse(self, daily_returns, deterministic_regimes):
        from src.risk_metrics import conditional_var
        var_df = conditional_var(daily_returns, deterministic_regimes)
        # Crisis VaR should be more negative than Bull VaR
        assert var_df.loc["Crisis", "VaR"] < var_df.loc["Bull", "VaR"]


class TestConditionalCVaR:
    def test_cvar_worse_than_var(self, daily_returns, deterministic_regimes):
        from src.risk_metrics import conditional_var, conditional_cvar
        var_df = conditional_var(daily_returns, deterministic_regimes)
        cvar_df = conditional_cvar(daily_returns, deterministic_regimes)
        for regime in var_df.index:
            assert cvar_df.loc[regime, "CVaR"] <= var_df.loc[regime, "VaR"]

    def test_has_all_regimes(self, daily_returns, deterministic_regimes):
        from src.risk_metrics import conditional_cvar
        cvar_df = conditional_cvar(daily_returns, deterministic_regimes)
        assert "Bull" in cvar_df.index
        assert "Crisis" in cvar_df.index


class TestConditionalVol:
    def test_crisis_vol_highest(self, daily_returns, deterministic_regimes):
        from src.risk_metrics import conditional_vol
        vol_df = conditional_vol(daily_returns, deterministic_regimes)
        assert vol_df.loc["Crisis", "vol"] > vol_df.loc["Bull", "vol"]

    def test_positive(self, daily_returns, deterministic_regimes):
        from src.risk_metrics import conditional_vol
        vol_df = conditional_vol(daily_returns, deterministic_regimes)
        assert (vol_df["vol"] > 0).all()


class TestRegimeSummaryTable:
    def test_days_sum_to_total(self, daily_returns, daily_prices, deterministic_regimes):
        from src.risk_metrics import regime_summary_table
        summary = regime_summary_table(daily_returns, daily_prices, deterministic_regimes)
        assert summary["days"].sum() == len(daily_returns)

    def test_proportion_sums_to_one(self, daily_returns, daily_prices, deterministic_regimes):
        from src.risk_metrics import regime_summary_table
        summary = regime_summary_table(daily_returns, daily_prices, deterministic_regimes)
        assert abs(summary["proportion"].sum() - 1.0) < 1e-10

    def test_max_drawdown_negative(self, daily_returns, daily_prices, deterministic_regimes):
        from src.risk_metrics import regime_summary_table
        summary = regime_summary_table(daily_returns, daily_prices, deterministic_regimes)
        assert (summary["max_drawdown"] <= 0).all()

    def test_crisis_vol_higher(self, daily_returns, daily_prices, deterministic_regimes):
        from src.risk_metrics import regime_summary_table
        summary = regime_summary_table(daily_returns, daily_prices, deterministic_regimes)
        assert summary.loc["Crisis", "ann_vol"] > summary.loc["Bull", "ann_vol"]


# ---------------------------------------------------------------------------
# Charts Tests (verify they produce figures without error)
# ---------------------------------------------------------------------------

class TestCharts:
    def test_regime_timeline(self, daily_prices, deterministic_regimes):
        from src.charts import plot_regime_timeline
        fig = plot_regime_timeline(daily_prices, deterministic_regimes)
        assert fig is not None

    def test_factor_heatmaps(self, factor_returns, deterministic_regimes):
        from src.factor_analysis import correlation_by_regime
        from src.charts import plot_factor_heatmaps
        corrs = correlation_by_regime(factor_returns, deterministic_regimes)
        fig = plot_factor_heatmaps(corrs)
        assert fig is not None

    def test_regime_distributions(self, daily_returns, deterministic_regimes):
        from src.charts import plot_regime_distributions
        fig = plot_regime_distributions(daily_returns, deterministic_regimes)
        assert fig is not None

    def test_transition_matrix(self, hmm_features):
        from src.regime_model import fit_hmm, transition_matrix
        from src.charts import plot_transition_matrix
        model = fit_hmm(hmm_features, n_states=3)
        tm = transition_matrix(model)
        fig = plot_transition_matrix(tm)
        assert fig is not None
