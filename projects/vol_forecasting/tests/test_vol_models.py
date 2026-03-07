"""
Tests for volatility forecasting project.

All tests use synthetic data — no network calls, runs instantly.
Covers: realized_vol, models (EWMA, GARCH, HAR-RV, VIX), evaluate, vol_target.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures — synthetic data mimicking daily stock returns
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_dates():
    """Two years of trading days (504)."""
    return pd.bdate_range("2022-01-01", periods=504, freq="B")


@pytest.fixture
def daily_returns(daily_dates):
    """Synthetic daily returns with known properties."""
    np.random.seed(42)
    # Regime switch: first half calm (low vol), second half volatile
    n = len(daily_dates)
    half = n // 2
    calm = np.random.normal(0.0005, 0.01, half)
    volatile = np.random.normal(-0.0002, 0.025, n - half)
    returns = np.concatenate([calm, volatile])
    return pd.Series(returns, index=daily_dates, name="returns")


@pytest.fixture
def daily_prices(daily_returns):
    """Synthetic daily prices from cumulative returns."""
    prices = 100 * (1 + daily_returns).cumprod()
    prices.name = "price"
    return prices


@pytest.fixture
def synthetic_vix(daily_dates):
    """Synthetic VIX in percentage terms (e.g., 15-30)."""
    np.random.seed(123)
    n = len(daily_dates)
    half = n // 2
    calm_vix = np.random.normal(15, 2, half).clip(10, 25)
    vol_vix = np.random.normal(28, 5, n - half).clip(15, 50)
    vix = np.concatenate([calm_vix, vol_vix])
    return pd.Series(vix, index=daily_dates, name="VIX")


@pytest.fixture
def universe_prices(daily_dates):
    """Synthetic daily prices for 10 stocks."""
    np.random.seed(99)
    n = len(daily_dates)
    tickers = [f"S{i}" for i in range(10)]
    data = {}
    for t in tickers:
        drift = np.random.normal(0.0003, 0.0001)
        vol = np.random.uniform(0.01, 0.03)
        data[t] = 100 * np.cumprod(1 + np.random.normal(drift, vol, n))
    return pd.DataFrame(data, index=daily_dates)


# ---------------------------------------------------------------------------
# Realized Vol Tests
# ---------------------------------------------------------------------------

class TestRealizedVol:
    def test_output_length(self, daily_returns):
        from src.realized_vol import realized_vol
        rv = realized_vol(daily_returns, window=22)
        assert len(rv) == len(daily_returns)

    def test_non_negative(self, daily_returns):
        from src.realized_vol import realized_vol
        rv = realized_vol(daily_returns, window=22)
        assert (rv.dropna() >= 0).all()

    def test_annualized(self, daily_returns):
        from src.realized_vol import realized_vol
        rv = realized_vol(daily_returns, window=22)
        raw_std = daily_returns.rolling(22).std()
        ratio = (rv / raw_std).dropna()
        assert np.allclose(ratio.values, np.sqrt(252), atol=0.01)

    def test_volatile_period_higher(self, daily_returns):
        from src.realized_vol import realized_vol
        rv = realized_vol(daily_returns, window=22)
        half = len(rv) // 2
        valid = rv.dropna()
        # Second half (volatile) should have higher mean RV
        calm_mean = valid.iloc[:half // 2].mean()
        vol_mean = valid.iloc[-half // 2:].mean()
        assert vol_mean > calm_mean


class TestForwardRV:
    def test_output_length(self, daily_returns):
        from src.realized_vol import forward_rv
        fwd = forward_rv(daily_returns, window=22)
        assert len(fwd) == len(daily_returns)

    def test_has_trailing_nans(self, daily_returns):
        from src.realized_vol import forward_rv
        fwd = forward_rv(daily_returns, window=22)
        # Last 22 values should be NaN (no future data)
        assert fwd.iloc[-22:].isna().all()

    def test_non_negative(self, daily_returns):
        from src.realized_vol import forward_rv
        fwd = forward_rv(daily_returns, window=22)
        assert (fwd.dropna() >= 0).all()


class TestHARFeatures:
    def test_output_columns(self, daily_returns):
        from src.realized_vol import har_features
        features = har_features(daily_returns)
        assert set(features.columns) == {"rv_d", "rv_w", "rv_m"}

    def test_output_length(self, daily_returns):
        from src.realized_vol import har_features
        features = har_features(daily_returns)
        assert len(features) == len(daily_returns)

    def test_daily_ge_zero(self, daily_returns):
        from src.realized_vol import har_features
        features = har_features(daily_returns)
        assert (features["rv_d"].dropna() >= 0).all()

    def test_weekly_smoother_than_daily(self, daily_returns):
        from src.realized_vol import har_features
        features = har_features(daily_returns)
        valid = features.dropna()
        # Weekly RV should be smoother (lower std) than daily RV
        assert valid["rv_w"].std() < valid["rv_d"].std()


class TestVolCone:
    def test_output_shape(self, daily_returns):
        from src.realized_vol import vol_cone
        cone = vol_cone(daily_returns)
        assert "50th" in cone.columns
        assert len(cone) > 0

    def test_percentile_ordering(self, daily_returns):
        from src.realized_vol import vol_cone
        cone = vol_cone(daily_returns)
        for _, row in cone.iterrows():
            assert row["10th"] <= row["25th"] <= row["50th"] <= row["75th"] <= row["90th"]


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------

class TestEWMA:
    def test_output_length(self, daily_returns):
        from src.models import ewma_vol
        vol = ewma_vol(daily_returns)
        assert len(vol) == len(daily_returns)

    def test_non_negative(self, daily_returns):
        from src.models import ewma_vol
        vol = ewma_vol(daily_returns)
        assert (vol.dropna() >= 0).all()

    def test_lambda_sensitivity(self, daily_returns):
        from src.models import ewma_vol
        # Higher lambda = smoother, lower lambda = more reactive
        vol_high = ewma_vol(daily_returns, lam=0.97)
        vol_low = ewma_vol(daily_returns, lam=0.90)
        valid_high = vol_high.dropna()
        valid_low = vol_low.dropna()
        # Lower lambda should have higher variance (more reactive)
        assert valid_low.std() > valid_high.std()

    def test_annualized(self, daily_returns):
        from src.models import ewma_vol
        vol = ewma_vol(daily_returns)
        # Annualized vol should typically be in 0.05-0.60 range for stock returns
        valid = vol.dropna()
        assert valid.median() > 0.01
        assert valid.median() < 1.0


class TestGARCH:
    def test_output_length(self, daily_returns):
        from src.models import garch_vol
        vol = garch_vol(daily_returns, min_history=252, refit_every=63)
        assert len(vol) == len(daily_returns)

    def test_non_negative(self, daily_returns):
        from src.models import garch_vol
        vol = garch_vol(daily_returns, min_history=252, refit_every=63)
        assert (vol.dropna() >= 0).all()

    def test_has_initial_nans(self, daily_returns):
        from src.models import garch_vol
        vol = garch_vol(daily_returns, min_history=252, refit_every=63)
        # First min_history values should be NaN
        assert vol.iloc[:252].isna().all()


class TestHARRV:
    def test_output_length(self, daily_returns):
        from src.models import har_rv_vol
        vol = har_rv_vol(daily_returns, min_history=252, refit_every=63)
        assert len(vol) == len(daily_returns)

    def test_non_negative(self, daily_returns):
        from src.models import har_rv_vol
        vol = har_rv_vol(daily_returns, min_history=252, refit_every=63)
        valid = vol.dropna()
        assert (valid >= 0).all()


class TestVIXImplied:
    def test_conversion(self, synthetic_vix):
        from src.models import vix_implied_vol
        vol = vix_implied_vol(synthetic_vix)
        # VIX of 20 should become 0.20
        assert np.allclose(vol.values, synthetic_vix.values / 100)

    def test_output_length(self, synthetic_vix):
        from src.models import vix_implied_vol
        vol = vix_implied_vol(synthetic_vix)
        assert len(vol) == len(synthetic_vix)


class TestRunAllForecasts:
    def test_output_columns(self, daily_returns, synthetic_vix):
        from src.models import run_all_forecasts
        forecasts = run_all_forecasts(daily_returns, synthetic_vix)
        expected = {"realized", "ewma", "garch", "har", "vix"}
        assert expected == set(forecasts.columns)

    def test_no_all_nan_rows(self, daily_returns, synthetic_vix):
        from src.models import run_all_forecasts
        forecasts = run_all_forecasts(daily_returns, synthetic_vix)
        assert not forecasts.isna().all(axis=1).any()


# ---------------------------------------------------------------------------
# Evaluate Tests
# ---------------------------------------------------------------------------

class TestQLIKE:
    def test_perfect_forecast(self):
        from src.evaluate import qlike
        realized = pd.Series([0.10, 0.20, 0.15, 0.25])
        forecast = pd.Series([0.10, 0.20, 0.15, 0.25])
        assert abs(qlike(realized, forecast)) < 1e-10

    def test_positive(self):
        from src.evaluate import qlike
        realized = pd.Series([0.10, 0.20, 0.15, 0.25])
        forecast = pd.Series([0.12, 0.18, 0.20, 0.22])
        assert qlike(realized, forecast) > 0

    def test_worse_forecast_higher_qlike(self):
        from src.evaluate import qlike
        realized = pd.Series([0.10, 0.20, 0.15, 0.25])
        good = pd.Series([0.11, 0.19, 0.14, 0.24])
        bad = pd.Series([0.05, 0.35, 0.08, 0.40])
        assert qlike(realized, bad) > qlike(realized, good)


class TestMSE:
    def test_perfect(self):
        from src.evaluate import mse
        s = pd.Series([0.1, 0.2, 0.3])
        assert abs(mse(s, s)) < 1e-10

    def test_positive(self):
        from src.evaluate import mse
        realized = pd.Series([0.10, 0.20, 0.15])
        forecast = pd.Series([0.12, 0.18, 0.20])
        assert mse(realized, forecast) > 0


class TestMAE:
    def test_perfect(self):
        from src.evaluate import mae
        s = pd.Series([0.1, 0.2, 0.3])
        assert abs(mae(s, s)) < 1e-10

    def test_known_value(self):
        from src.evaluate import mae
        realized = pd.Series([0.10, 0.20])
        forecast = pd.Series([0.12, 0.18])
        assert abs(mae(realized, forecast) - 0.02) < 1e-10


class TestMincerZarnowitz:
    def test_perfect_forecast(self, capsys):
        from src.evaluate import mincer_zarnowitz
        np.random.seed(42)
        true_vol = pd.Series(np.random.uniform(0.10, 0.30, 100))
        result = mincer_zarnowitz(true_vol, true_vol, name="Perfect")
        assert abs(result["alpha"]) < 1e-10
        assert abs(result["beta"] - 1.0) < 1e-10
        assert result["r_squared"] > 0.999


# ---------------------------------------------------------------------------
# Vol Target Tests
# ---------------------------------------------------------------------------

class TestEqualWeightReturns:
    def test_output_is_series(self, universe_prices):
        from src.vol_target import equal_weight_returns
        ret = equal_weight_returns(universe_prices)
        assert isinstance(ret, pd.Series)
        assert len(ret) > 0

    def test_reasonable_returns(self, universe_prices):
        from src.vol_target import equal_weight_returns
        ret = equal_weight_returns(universe_prices)
        # Daily returns should be small
        assert abs(ret.mean()) < 0.05
        assert ret.std() < 0.10


@pytest.fixture
def daily_returns_as_vol_forecast(daily_returns):
    """Use rolling std as a simple vol forecast for testing."""
    return daily_returns.rolling(22).std() * np.sqrt(252)


class TestVolTargetedReturns:
    def test_output_length(self, daily_returns, daily_returns_as_vol_forecast):
        from src.vol_target import vol_targeted_returns
        managed = vol_targeted_returns(daily_returns, daily_returns_as_vol_forecast)
        assert len(managed) > 0

    def test_leverage_clipping(self):
        from src.vol_target import vol_targeted_returns
        dates = pd.bdate_range("2023-01-01", periods=100, freq="B")
        ret = pd.Series(np.random.normal(0, 0.01, 100), index=dates)
        # Very low vol forecast → should hit max leverage
        low_vol = pd.Series(np.full(100, 0.01), index=dates)
        managed = vol_targeted_returns(ret, low_vol, target_vol=0.10, max_leverage=2.0)
        # Managed returns should be at most 2x the raw returns (plus noise from shift)
        assert managed.dropna().abs().max() < ret.abs().max() * 2.5


class TestComputePerformance:
    def test_known_sharpe(self):
        from src.vol_target import compute_performance
        # Constant daily return of 0.1% with no volatility → infinite Sharpe
        # Use small but nonzero vol
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.005, 252))
        metrics = compute_performance(returns, rf=0.0)
        assert "sharpe" in metrics
        assert "max_drawdown" in metrics
        assert metrics["max_drawdown"] <= 0

    def test_empty_returns(self):
        from src.vol_target import compute_performance
        returns = pd.Series(dtype=float)
        metrics = compute_performance(returns)
        assert metrics == {}
