"""
Tests for sharpe_101.py — core Sharpe ratio functions.

All tests use synthetic/deterministic data. No network calls.
"""

import numpy as np
import pandas as pd
import pytest

from sharpe_101 import (
    simple_returns,
    log_returns,
    excess_returns,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    rolling_sharpe,
    TRADING_DAYS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prices():
    """Simple deterministic price series: 100, 102, 101, 105, 103."""
    return pd.Series([100.0, 102.0, 101.0, 105.0, 103.0],
                     index=pd.date_range("2020-01-01", periods=5))


@pytest.fixture
def constant_prices():
    """Constant prices — zero volatility case."""
    return pd.Series([100.0] * 10,
                     index=pd.date_range("2020-01-01", periods=10))


@pytest.fixture
def long_returns():
    """A longer return series for rolling tests."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.01, 200),
                     index=pd.date_range("2020-01-01", periods=200))


# ---------------------------------------------------------------------------
# simple_returns
# ---------------------------------------------------------------------------

class TestSimpleReturns:
    def test_basic(self, prices):
        result = simple_returns(prices)
        expected = pd.Series(
            [0.02, -1 / 102, 4 / 101, -2 / 105],
            index=prices.index[1:],
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_length(self, prices):
        result = simple_returns(prices)
        assert len(result) == len(prices) - 1

    def test_constant_prices(self, constant_prices):
        result = simple_returns(constant_prices)
        assert (result == 0).all()

    def test_two_prices(self):
        prices = pd.Series([100.0, 110.0])
        result = simple_returns(prices)
        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# log_returns
# ---------------------------------------------------------------------------

class TestLogReturns:
    def test_basic(self, prices):
        result = log_returns(prices)
        expected = np.log(prices / prices.shift(1)).dropna()
        pd.testing.assert_series_equal(result, expected)

    def test_length(self, prices):
        assert len(log_returns(prices)) == len(prices) - 1

    def test_close_to_simple_for_small_returns(self, prices):
        sr = simple_returns(prices)
        lr = log_returns(prices)
        # For small returns, log and simple are approximately equal
        np.testing.assert_allclose(lr.values, sr.values, atol=0.005)

    def test_constant_prices(self, constant_prices):
        result = log_returns(constant_prices)
        assert (result == 0).all()


# ---------------------------------------------------------------------------
# excess_returns
# ---------------------------------------------------------------------------

class TestExcessReturns:
    def test_basic(self):
        returns = pd.Series([0.01, 0.02, -0.01, 0.005])
        annual_rf = 0.0252  # makes daily_rf = 0.0001
        result = excess_returns(returns, annual_rf)
        daily_rf = 0.0252 / 252
        expected = returns - daily_rf
        pd.testing.assert_series_equal(result, expected)

    def test_zero_rf(self):
        returns = pd.Series([0.01, 0.02])
        result = excess_returns(returns, 0.0)
        pd.testing.assert_series_equal(result, returns)

    def test_rf_subtracted_correctly(self):
        returns = pd.Series([0.001])
        annual_rf = 0.252  # daily = 0.001
        result = excess_returns(returns, annual_rf)
        assert result.iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------

class TestAnnualizedReturn:
    def test_basic(self):
        daily = pd.Series([0.001] * 252)
        result = annualized_return(daily)
        assert result == pytest.approx(0.001 * 252)

    def test_zero_returns(self):
        daily = pd.Series([0.0] * 100)
        assert annualized_return(daily) == pytest.approx(0.0)

    def test_negative_returns(self):
        daily = pd.Series([-0.001] * 100)
        assert annualized_return(daily) == pytest.approx(-0.001 * 252)


# ---------------------------------------------------------------------------
# annualized_volatility
# ---------------------------------------------------------------------------

class TestAnnualizedVolatility:
    def test_basic(self):
        daily = pd.Series([0.01, -0.01, 0.01, -0.01] * 50)
        daily_std = daily.std()
        result = annualized_volatility(daily)
        assert result == pytest.approx(daily_std * np.sqrt(252))

    def test_constant_returns(self):
        daily = pd.Series([0.001] * 100)
        assert annualized_volatility(daily) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_zero_vol_exact(self):
        # When excess returns have exactly zero std, sharpe_ratio returns 0.0
        # Use rf equal to the return so excess is exactly 0
        daily = pd.Series([0.001] * 252)
        # annual_rf = 0.001 * 252 makes daily_rf = 0.001, excess = 0
        result = sharpe_ratio(daily, annual_rf=0.001 * 252)
        assert result == 0.0

    def test_positive_sharpe(self):
        np.random.seed(123)
        daily = pd.Series(np.random.normal(0.001, 0.01, 500))
        result = sharpe_ratio(daily, annual_rf=0.0)
        assert result > 0

    def test_zero_volatility_returns_zero(self):
        daily = pd.Series([0.001] * 50)
        result = sharpe_ratio(daily, annual_rf=0.0)
        assert result == 0.0

    def test_manual_calculation(self):
        np.random.seed(42)
        daily = pd.Series(np.random.normal(0.001, 0.02, 500))
        annual_rf = 0.05
        # Manual computation
        daily_rf = annual_rf / 252
        excess = daily - daily_rf
        ann_ret = excess.mean() * 252
        ann_vol = excess.std() * np.sqrt(252)
        expected = ann_ret / ann_vol
        assert sharpe_ratio(daily, annual_rf) == pytest.approx(expected)

    def test_all_negative_returns(self):
        daily = pd.Series([-0.01, -0.02, -0.005, -0.015, -0.01])
        result = sharpe_ratio(daily, annual_rf=0.0)
        assert result < 0


# ---------------------------------------------------------------------------
# rolling_sharpe
# ---------------------------------------------------------------------------

class TestRollingSharpe:
    def test_output_length(self, long_returns):
        window = 63
        result = rolling_sharpe(long_returns, window=window, annual_rf=0.05)
        # After rolling + dropna, length = len - window + 1 (from rolling)
        # then dropna removes NaN entries
        assert len(result) == len(long_returns) - window + 1

    def test_window_larger_than_data(self):
        short = pd.Series([0.01, -0.01, 0.02])
        result = rolling_sharpe(short, window=10)
        assert len(result) == 0

    def test_values_are_finite(self, long_returns):
        result = rolling_sharpe(long_returns, window=20, annual_rf=0.05)
        assert result.notna().all()
        assert np.isfinite(result).all()

    def test_returns_series(self, long_returns):
        result = rolling_sharpe(long_returns, window=20)
        assert isinstance(result, pd.Series)
