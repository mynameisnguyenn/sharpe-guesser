"""
Tests for modules 2, 3, 4, and 5.

All tests use synthetic/deterministic data. No network calls.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from modules.module_2_risk_metrics import (
    var_historical,
    var_parametric,
    cvar,
    max_drawdown,
    drawdown_series,
    downside_deviation,
    sortino_ratio,
    calmar_ratio,
)
from modules.module_3_factor_models import (
    capm_regression,
    rolling_beta,
    information_ratio,
)
from modules.module_4_portfolio_optimisation import (
    portfolio_return,
    portfolio_volatility,
    portfolio_sharpe,
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
    risk_contribution,
    risk_parity_portfolio,
)
from modules.module_5_strategies import SimpleBacktester


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_returns():
    """Deterministic daily returns for risk metric tests."""
    np.random.seed(42)
    return pd.Series(
        np.random.normal(0.0005, 0.015, 500),
        index=pd.date_range("2020-01-01", periods=500, freq="B"),
    )


@pytest.fixture
def known_prices():
    """Price series with a known drawdown pattern: 100 -> 120 -> 90 -> 110."""
    return pd.Series(
        [100.0, 110.0, 120.0, 100.0, 90.0, 95.0, 110.0, 115.0],
        index=pd.date_range("2020-01-01", periods=8),
    )


@pytest.fixture
def stock_and_market():
    """Synthetic stock and market returns with known beta."""
    np.random.seed(42)
    n = 500
    market = pd.Series(
        np.random.normal(0.0005, 0.01, n),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
        name="market",
    )
    # stock = alpha + beta * market + noise
    alpha_daily = 0.0002
    beta = 1.5
    noise = np.random.normal(0, 0.005, n)
    stock = pd.Series(
        alpha_daily + beta * market.values + noise,
        index=market.index,
        name="stock",
    )
    return stock, market, beta, alpha_daily


# ---------------------------------------------------------------------------
# Module 2: VaR
# ---------------------------------------------------------------------------

class TestVarHistorical:
    def test_95_confidence(self, daily_returns):
        result = var_historical(daily_returns, 0.95)
        expected = daily_returns.quantile(0.05)
        assert result == pytest.approx(expected)

    def test_99_confidence(self, daily_returns):
        result = var_historical(daily_returns, 0.99)
        expected = daily_returns.quantile(0.01)
        assert result == pytest.approx(expected)

    def test_var_is_negative_for_normal_returns(self, daily_returns):
        result = var_historical(daily_returns, 0.95)
        assert result < 0

    def test_higher_confidence_more_extreme(self, daily_returns):
        var_95 = var_historical(daily_returns, 0.95)
        var_99 = var_historical(daily_returns, 0.99)
        assert var_99 < var_95


class TestVarParametric:
    def test_matches_scipy(self, daily_returns):
        mu = daily_returns.mean()
        sigma = daily_returns.std()
        result = var_parametric(daily_returns, 0.95)
        expected = sp_stats.norm.ppf(0.05, mu, sigma)
        assert result == pytest.approx(expected)

    def test_higher_confidence_more_extreme(self, daily_returns):
        var_95 = var_parametric(daily_returns, 0.95)
        var_99 = var_parametric(daily_returns, 0.99)
        assert var_99 < var_95


# ---------------------------------------------------------------------------
# Module 2: CVaR
# ---------------------------------------------------------------------------

class TestCVaR:
    def test_cvar_below_var(self, daily_returns):
        var_val = var_historical(daily_returns, 0.95)
        cvar_val = cvar(daily_returns, 0.95)
        assert cvar_val <= var_val

    def test_cvar_is_mean_of_tail(self, daily_returns):
        var_val = var_historical(daily_returns, 0.95)
        tail = daily_returns[daily_returns <= var_val]
        expected = tail.mean()
        result = cvar(daily_returns, 0.95)
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Module 2: Drawdowns
# ---------------------------------------------------------------------------

class TestDrawdownSeries:
    def test_known_drawdown(self, known_prices):
        dd = drawdown_series(known_prices)
        # At peak (120), drawdown = 0
        assert dd.iloc[2] == pytest.approx(0.0)
        # At 90, drawdown from peak of 120 = (90-120)/120 = -0.25
        assert dd.iloc[4] == pytest.approx(-0.25)
        # First price is peak at that point, dd = 0
        assert dd.iloc[0] == pytest.approx(0.0)

    def test_all_positive_starts_at_zero(self, known_prices):
        dd = drawdown_series(known_prices)
        assert dd.iloc[0] == 0.0

    def test_never_positive(self, known_prices):
        dd = drawdown_series(known_prices)
        assert (dd <= 0).all()


class TestMaxDrawdown:
    def test_known_max_drawdown(self, known_prices):
        # Max drawdown is from 120 to 90 = -25%
        result = max_drawdown(known_prices)
        assert result == pytest.approx(-0.25)

    def test_monotonically_increasing_prices(self):
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        result = max_drawdown(prices)
        assert result == pytest.approx(0.0)

    def test_monotonically_decreasing_prices(self):
        prices = pd.Series([100.0, 90.0, 80.0, 70.0])
        result = max_drawdown(prices)
        # Max dd from 100 to 70 = -30%
        assert result == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# Module 2: Sortino Ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_positive_for_positive_mean(self, daily_returns):
        result = sortino_ratio(daily_returns, annual_rf=0.0)
        assert result > 0

    def test_zero_downside_returns_zero(self):
        # All positive returns => no negative excess returns => downside std is NaN
        # sortino_ratio guards against NaN downside deviation, returning 0.0.
        rets = pd.Series([0.01, 0.02, 0.015, 0.005, 0.01])
        result = sortino_ratio(rets, annual_rf=0.0)
        assert result == 0.0

    def test_all_negative_returns(self):
        rets = pd.Series([-0.01, -0.02, -0.005, -0.015, -0.01])
        result = sortino_ratio(rets, annual_rf=0.0)
        assert result < 0

    def test_sortino_gte_sharpe_for_positive_mean(self):
        # With symmetric returns and positive mean, downside vol < total vol,
        # so Sortino should exceed Sharpe
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0.001, 0.01, 500))
        from sharpe_101 import sharpe_ratio
        sr = sharpe_ratio(rets, annual_rf=0.0)
        so = sortino_ratio(rets, annual_rf=0.0)
        assert so >= sr


# ---------------------------------------------------------------------------
# Module 2: Calmar Ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_known_value(self, known_prices):
        result = calmar_ratio(known_prices, annual_rf=0.0)
        # Returns from known_prices
        rets = known_prices.pct_change().dropna()
        ann_excess = rets.mean() * 252
        mdd = abs(max_drawdown(known_prices))
        expected = ann_excess / mdd
        assert result == pytest.approx(expected)

    def test_no_drawdown_returns_zero(self):
        prices = pd.Series([100.0, 101.0, 102.0, 103.0])
        result = calmar_ratio(prices, annual_rf=0.0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Module 3: CAPM Regression
# ---------------------------------------------------------------------------

class TestCAPMRegression:
    def test_beta_recovery(self, stock_and_market):
        stock, market, true_beta, _ = stock_and_market
        result = capm_regression(stock, market, annual_rf=0.0)
        assert result["beta"] == pytest.approx(true_beta, abs=0.15)

    def test_r_squared_high(self, stock_and_market):
        stock, market, _, _ = stock_and_market
        result = capm_regression(stock, market, annual_rf=0.0)
        assert result["r_squared"] > 0.5

    def test_returns_dict_keys(self, stock_and_market):
        stock, market, _, _ = stock_and_market
        result = capm_regression(stock, market)
        expected_keys = {"alpha_daily", "alpha_annual", "beta", "r_squared",
                         "alpha_tstat", "beta_tstat", "model"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Module 3: Rolling Beta
# ---------------------------------------------------------------------------

class TestRollingBeta:
    def test_output_length(self, stock_and_market):
        stock, market, _, _ = stock_and_market
        window = 63
        result = rolling_beta(stock, market, window=window)
        # After alignment, rolling, and dropna
        assert len(result) > 0
        assert len(result) <= len(stock) - window + 1

    def test_mean_close_to_true_beta(self, stock_and_market):
        stock, market, true_beta, _ = stock_and_market
        result = rolling_beta(stock, market, window=63)
        assert result.mean() == pytest.approx(true_beta, abs=0.3)

    def test_returns_series(self, stock_and_market):
        stock, market, _, _ = stock_and_market
        result = rolling_beta(stock, market, window=20)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Module 3: Information Ratio
# ---------------------------------------------------------------------------

class TestInformationRatio:
    def test_positive_alpha_positive_ir(self, stock_and_market):
        stock, market, _, _ = stock_and_market
        result = capm_regression(stock, market, annual_rf=0.0)
        ir = information_ratio(result)
        # We injected positive alpha, so IR should be positive
        assert ir > 0

    def test_returns_float(self, stock_and_market):
        stock, market, _, _ = stock_and_market
        result = capm_regression(stock, market)
        ir = information_ratio(result)
        assert isinstance(ir, float)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_return_sharpe(self):
        daily = pd.Series([0.01])
        from sharpe_101 import sharpe_ratio
        result = sharpe_ratio(daily, annual_rf=0.0)
        # std of single element is NaN with ddof=1 => vol = NaN => 0
        assert result == 0.0 or np.isnan(result)

    def test_nan_in_returns(self):
        daily = pd.Series([0.01, np.nan, -0.01, 0.02, np.nan, 0.005])
        result = var_historical(daily.dropna(), 0.95)
        assert np.isfinite(result)

    def test_empty_series_var(self):
        daily = pd.Series([], dtype=float)
        # quantile on empty series returns NaN
        result = var_historical(daily, 0.95)
        assert np.isnan(result)

    def test_all_same_returns(self):
        daily = pd.Series([0.01] * 100)
        result = var_historical(daily, 0.95)
        assert result == pytest.approx(0.01)

    def test_drawdown_single_price(self):
        prices = pd.Series([100.0])
        dd = drawdown_series(prices)
        assert dd.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Module 4: Portfolio Math
# ---------------------------------------------------------------------------

@pytest.fixture
def portfolio_data():
    """Synthetic expected returns and covariance for 3 assets."""
    mu = pd.Series([0.10, 0.15, 0.08], index=["A", "B", "C"])
    # Build a valid positive-definite covariance matrix
    cov = pd.DataFrame(
        [[0.04, 0.006, 0.002],
         [0.006, 0.09, 0.004],
         [0.002, 0.004, 0.01]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    return mu, cov


class TestPortfolioReturn:
    def test_equal_weight(self, portfolio_data):
        mu, _ = portfolio_data
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = portfolio_return(w, mu)
        assert result == pytest.approx(mu.mean())

    def test_single_asset(self, portfolio_data):
        mu, _ = portfolio_data
        w = np.array([1.0, 0.0, 0.0])
        result = portfolio_return(w, mu)
        assert result == pytest.approx(0.10)


class TestPortfolioVolatility:
    def test_single_asset(self, portfolio_data):
        _, cov = portfolio_data
        w = np.array([1.0, 0.0, 0.0])
        result = portfolio_volatility(w, cov)
        assert result == pytest.approx(np.sqrt(0.04))

    def test_positive(self, portfolio_data):
        _, cov = portfolio_data
        w = np.array([0.5, 0.3, 0.2])
        result = portfolio_volatility(w, cov)
        assert result > 0

    def test_diversification_benefit(self, portfolio_data):
        _, cov = portfolio_data
        # Equal weight vol should be less than average individual vol
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        port_vol = portfolio_volatility(w, cov)
        avg_vol = np.mean([np.sqrt(cov.iloc[i, i]) for i in range(3)])
        assert port_vol < avg_vol


class TestPortfolioSharpe:
    def test_zero_vol_returns_zero(self):
        mu = pd.Series([0.05])
        cov = pd.DataFrame([[0.0]], index=["A"], columns=["A"])
        w = np.array([1.0])
        result = portfolio_sharpe(w, mu, cov, rf=0.05)
        assert result == 0.0

    def test_positive_excess_return(self, portfolio_data):
        mu, cov = portfolio_data
        w = np.array([0.4, 0.4, 0.2])
        result = portfolio_sharpe(w, mu, cov, rf=0.05)
        assert result > 0


class TestMinimumVariancePortfolio:
    def test_returns_dict_keys(self, portfolio_data):
        mu, cov = portfolio_data
        result = minimum_variance_portfolio(mu, cov)
        assert set(result.keys()) == {"weights", "return", "volatility", "sharpe"}

    def test_weights_sum_to_one(self, portfolio_data):
        mu, cov = portfolio_data
        result = minimum_variance_portfolio(mu, cov)
        assert np.sum(result["weights"]) == pytest.approx(1.0)

    def test_no_short_weights(self, portfolio_data):
        mu, cov = portfolio_data
        result = minimum_variance_portfolio(mu, cov, allow_short=False)
        assert all(w >= -1e-10 for w in result["weights"])

    def test_lower_vol_than_equal_weight(self, portfolio_data):
        mu, cov = portfolio_data
        mv = minimum_variance_portfolio(mu, cov)
        ew_vol = portfolio_volatility(np.ones(3) / 3, cov)
        assert mv["volatility"] <= ew_vol + 1e-8


class TestMaximumSharpePortfolio:
    def test_weights_sum_to_one(self, portfolio_data):
        mu, cov = portfolio_data
        result = maximum_sharpe_portfolio(mu, cov, rf=0.05)
        assert np.sum(result["weights"]) == pytest.approx(1.0)

    def test_higher_sharpe_than_equal_weight(self, portfolio_data):
        mu, cov = portfolio_data
        ms = maximum_sharpe_portfolio(mu, cov, rf=0.05)
        ew_sharpe = portfolio_sharpe(np.ones(3) / 3, mu, cov, rf=0.05)
        assert ms["sharpe"] >= ew_sharpe - 1e-6


class TestRiskParity:
    def test_weights_sum_to_one(self, portfolio_data):
        _, cov = portfolio_data
        result = risk_parity_portfolio(cov)
        assert np.sum(result["weights"]) == pytest.approx(1.0)

    def test_risk_contributions_roughly_equal(self, portfolio_data):
        _, cov = portfolio_data
        result = risk_parity_portfolio(cov)
        rc = result["risk_contributions"]
        target = 1.0 / len(cov)
        for c in rc:
            assert c == pytest.approx(target, abs=0.05)


# ---------------------------------------------------------------------------
# Module 5: SimpleBacktester
# ---------------------------------------------------------------------------

@pytest.fixture
def backtester_data():
    """Synthetic returns and positions for backtesting."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    returns = pd.DataFrame({
        "A": np.random.normal(0.001, 0.02, 100),
        "B": np.random.normal(0.0005, 0.015, 100),
    }, index=dates)
    positions = pd.DataFrame({
        "A": [0.5] * 100,
        "B": [0.5] * 100,
    }, index=dates)
    return returns, positions


class TestSimpleBacktester:
    def test_stats_keys(self, backtester_data):
        returns, positions = backtester_data
        bt = SimpleBacktester(returns, positions, name="Test")
        s = bt.stats()
        expected_keys = {
            "annual_return", "annual_vol", "sharpe", "sortino",
            "calmar", "max_drawdown", "win_rate", "annual_turnover",
            "total_return",
        }
        assert set(s.keys()) == expected_keys

    def test_cumulative_starts_near_one(self, backtester_data):
        returns, positions = backtester_data
        bt = SimpleBacktester(returns, positions, name="Test")
        # First cumulative value should be near 1 (first day return applied)
        assert bt.cumulative.iloc[0] == pytest.approx(1.0, abs=0.05)

    def test_zero_transaction_costs(self, backtester_data):
        returns, positions = backtester_data
        bt_zero = SimpleBacktester(returns, positions, transaction_cost_bps=0, name="Zero TC")
        bt_ten = SimpleBacktester(returns, positions, transaction_cost_bps=10, name="Ten TC")
        # Gross return >= net return
        assert bt_zero.stats()["total_return"] >= bt_ten.stats()["total_return"]

    def test_constant_positions_low_turnover(self, backtester_data):
        returns, positions = backtester_data
        bt = SimpleBacktester(returns, positions, name="Constant Pos")
        s = bt.stats()
        # Constant positions => turnover comes only from first diff (NaN->0.5)
        # After that, diffs are 0, so annual_turnover should be very low
        assert s["annual_turnover"] < 10

    def test_sharpe_is_finite(self, backtester_data):
        returns, positions = backtester_data
        bt = SimpleBacktester(returns, positions, name="Test")
        s = bt.stats()
        assert np.isfinite(s["sharpe"])

    def test_max_drawdown_negative_or_zero(self, backtester_data):
        returns, positions = backtester_data
        bt = SimpleBacktester(returns, positions, name="Test")
        assert bt.stats()["max_drawdown"] <= 0
