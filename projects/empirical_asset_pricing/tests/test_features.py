"""
Tests for feature construction functions.

Uses synthetic data so tests run instantly without needing market data.
Each test validates the math, NaN handling, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    momentum_1m,
    momentum_6m,
    momentum_12m,
    realized_volatility,
    log_dollar_volume,
    volume_trend,
    max_return,
    build_features,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic data that mimics daily stock prices
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_dates():
    """One year of trading days (252)."""
    return pd.bdate_range("2023-01-01", periods=300, freq="B")


@pytest.fixture
def prices(daily_dates):
    """Synthetic daily prices for 3 stocks with known trends."""
    np.random.seed(42)
    n = len(daily_dates)

    # Stock A: uptrend, Stock B: downtrend, Stock C: flat
    data = {
        "A": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)),
        "B": 100 * np.cumprod(1 + np.random.normal(-0.001, 0.02, n)),
        "C": 100 * np.cumprod(1 + np.random.normal(0.0, 0.01, n)),
    }
    return pd.DataFrame(data, index=daily_dates)


@pytest.fixture
def volumes(daily_dates):
    """Synthetic daily volumes."""
    np.random.seed(123)
    n = len(daily_dates)
    data = {
        "A": np.random.randint(1_000_000, 10_000_000, n),
        "B": np.random.randint(500_000, 5_000_000, n),
        "C": np.random.randint(100_000, 1_000_000, n),
    }
    return pd.DataFrame(data, index=daily_dates)


@pytest.fixture
def returns(prices):
    """Daily returns computed from prices."""
    return prices.pct_change()


# ---------------------------------------------------------------------------
# Momentum tests
# ---------------------------------------------------------------------------

class TestMomentum1m:
    def test_output_shape(self, prices):
        result = momentum_1m(prices)
        assert result.shape == prices.shape

    def test_first_21_rows_are_nan(self, prices):
        result = momentum_1m(prices)
        assert result.iloc[:21].isna().all().all()

    def test_values_are_returns(self, prices):
        """1-month momentum should be (P_t / P_{t-21}) - 1."""
        result = momentum_1m(prices)
        expected = prices.iloc[21] / prices.iloc[0] - 1
        pd.testing.assert_series_equal(result.iloc[21], expected, check_names=False)


class TestMomentum6m:
    def test_output_shape(self, prices):
        result = momentum_6m(prices)
        assert result.shape == prices.shape

    def test_skips_most_recent_month(self, prices):
        """6m momentum should NOT include the most recent month's return."""
        mom_6m = momentum_6m(prices)
        mom_1m = momentum_1m(prices)

        # At a point where both are defined, 6m momentum should differ from
        # the raw 6-month return (because we skip the recent month)
        raw_6m = prices.pct_change(periods=126)
        idx = 150  # well past any NaN lookback

        # The skip-month version should differ from raw
        # (unless by coincidence, which is extremely unlikely with random data)
        assert not np.allclose(
            mom_6m.iloc[idx].values,
            raw_6m.iloc[idx].values,
            atol=1e-8,
        )

    def test_first_126_rows_are_nan(self, prices):
        result = momentum_6m(prices)
        assert result.iloc[:126].isna().all().all()


class TestMomentum12m:
    def test_output_shape(self, prices):
        result = momentum_12m(prices)
        assert result.shape == prices.shape

    def test_first_252_rows_are_nan(self, prices):
        result = momentum_12m(prices)
        assert result.iloc[:252].isna().all().all()

    def test_skips_most_recent_month(self, prices):
        """12m momentum should exclude the most recent month."""
        mom_12m = momentum_12m(prices)
        raw_12m = prices.pct_change(periods=252)

        # Only test where both are non-NaN
        valid_idx = mom_12m.first_valid_index()
        if valid_idx is not None:
            row_num = prices.index.get_loc(valid_idx)
            assert not np.allclose(
                mom_12m.iloc[row_num].values,
                raw_12m.iloc[row_num].values,
                atol=1e-8,
            )


# ---------------------------------------------------------------------------
# Volatility tests
# ---------------------------------------------------------------------------

class TestRealizedVolatility:
    def test_output_shape(self, returns):
        result = realized_volatility(returns, window=21)
        assert result.shape == returns.shape

    def test_non_negative(self, returns):
        result = realized_volatility(returns, window=21)
        assert (result.dropna() >= 0).all().all()

    def test_flat_stock_has_lower_vol(self, returns):
        """Stock C (flat) should have lower vol than A or B."""
        result = realized_volatility(returns, window=21)
        valid = result.dropna()
        assert valid["C"].mean() < valid["A"].mean()

    def test_annualization(self, returns):
        """Vol should be annualized (multiplied by sqrt(252))."""
        result = realized_volatility(returns, window=21)
        raw_std = returns.rolling(window=21).std()
        ratio = (result / raw_std).dropna()
        expected_ratio = np.sqrt(252)
        assert np.allclose(ratio.values, expected_ratio, atol=0.01)


# ---------------------------------------------------------------------------
# Size / volume tests
# ---------------------------------------------------------------------------

class TestLogDollarVolume:
    def test_output_shape(self, prices, volumes):
        result = log_dollar_volume(prices, volumes, window=21)
        assert result.shape == prices.shape

    def test_log_is_positive(self, prices, volumes):
        result = log_dollar_volume(prices, volumes, window=21)
        assert (result.dropna() > 0).all().all()

    def test_higher_volume_stock_has_larger_value(self, prices, volumes):
        """Stock A has highest volume, should have highest log dollar volume."""
        result = log_dollar_volume(prices, volumes, window=21)
        valid = result.dropna()
        assert valid["A"].mean() > valid["C"].mean()


class TestVolumeTrend:
    def test_output_shape(self, volumes):
        result = volume_trend(volumes, short=5, long=21)
        assert result.shape == volumes.shape

    def test_stable_volume_near_one(self):
        """Constant volume should give a trend ratio close to 1."""
        dates = pd.bdate_range("2023-01-01", periods=100, freq="B")
        constant_vol = pd.DataFrame(
            {"X": [1_000_000] * 100}, index=dates
        )
        result = volume_trend(constant_vol, short=5, long=21)
        valid = result.dropna()
        assert np.allclose(valid.values, 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# Max return test
# ---------------------------------------------------------------------------

class TestMaxReturn:
    def test_max_is_at_least_as_large_as_current(self, returns):
        """Max return over window should be >= each individual return."""
        result = max_return(returns, window=21)
        valid_idx = result.dropna().index
        assert (result.loc[valid_idx] >= returns.loc[valid_idx] - 1e-10).all().all()


# ---------------------------------------------------------------------------
# Build features integration test
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_returns_multiindex(self, prices, volumes):
        result = build_features(prices, volumes)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["date", "ticker"]

    def test_has_expected_columns(self, prices, volumes):
        result = build_features(prices, volumes)
        expected = {"mom_1m", "mom_6m", "mom_12m", "realized_vol",
                    "log_dollar_vol", "volume_trend", "max_return"}
        assert expected == set(result.columns)

    def test_no_all_nan_rows(self, prices, volumes):
        result = build_features(prices, volumes)
        assert not result.isna().all(axis=1).any()

    def test_output_not_empty(self, prices, volumes):
        result = build_features(prices, volumes)
        assert len(result) > 0
