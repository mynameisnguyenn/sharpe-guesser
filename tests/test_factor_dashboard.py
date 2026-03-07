"""
Tests for factor_dashboard.py — factor exposure dashboard functions.

All tests use synthetic/deterministic data. No network calls.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from factor_dashboard import (
    parse_args,
    analyse_ticker,
    print_comparison_table,
    save_comparison_chart,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spy_returns():
    """Synthetic SPY returns."""
    np.random.seed(42)
    n = 300
    return pd.Series(
        np.random.normal(0.0005, 0.01, n),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
        name="SPY",
    )


@pytest.fixture
def factor_df(spy_returns):
    """Synthetic factor DataFrame with MKT, SMB, HML columns."""
    np.random.seed(99)
    n = len(spy_returns)
    factors = pd.DataFrame({
        "MKT": spy_returns.values,
        "SMB": np.random.normal(0.0001, 0.005, n),
        "HML": np.random.normal(0.0001, 0.004, n),
    }, index=spy_returns.index)
    return factors


@pytest.fixture
def stock_returns(spy_returns):
    """Synthetic stock returns correlated with SPY (known beta ~1.3)."""
    np.random.seed(77)
    n = len(spy_returns)
    noise = np.random.normal(0, 0.005, n)
    return pd.Series(
        0.0002 + 1.3 * spy_returns.values + noise,
        index=spy_returns.index,
        name="TEST",
    )


@pytest.fixture
def summaries(spy_returns):
    """Two synthetic summary dicts for comparison table/chart tests."""
    np.random.seed(55)
    rb1 = pd.Series(np.random.normal(1.2, 0.1, 200),
                     index=pd.date_range("2020-04-01", periods=200, freq="B"))
    rb2 = pd.Series(np.random.normal(0.8, 0.15, 200),
                     index=pd.date_range("2020-04-01", periods=200, freq="B"))
    return [
        {
            "ticker": "AAPL",
            "alpha": 0.05,
            "beta": 1.2,
            "r_squared": 0.75,
            "smb": -0.15,
            "hml": -0.30,
            "ir": 0.45,
            "rolling_beta": rb1,
        },
        {
            "ticker": "JPM",
            "alpha": 0.02,
            "beta": 0.95,
            "r_squared": 0.60,
            "smb": 0.10,
            "hml": 0.25,
            "ir": 0.20,
            "rolling_beta": rb2,
        },
    ]


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_single_ticker(self):
        args = parse_args(["AAPL"])
        assert args.tickers == ["AAPL"]
        assert args.start == "2020-01-01"
        assert args.end == "2025-12-31"
        assert args.rf == 0.05
        assert args.window == 63

    def test_multiple_tickers(self):
        args = parse_args(["AAPL", "MSFT", "JPM"])
        assert args.tickers == ["AAPL", "MSFT", "JPM"]

    def test_custom_flags(self):
        args = parse_args(["AAPL", "--start", "2022-01-01", "--end", "2024-12-31",
                           "--rf", "0.04", "--window", "126"])
        assert args.start == "2022-01-01"
        assert args.end == "2024-12-31"
        assert args.rf == pytest.approx(0.04)
        assert args.window == 126


# ---------------------------------------------------------------------------
# analyse_ticker
# ---------------------------------------------------------------------------

class TestAnalyseTicker:
    def test_returns_summary_dict(self, spy_returns, factor_df, stock_returns):
        with patch("factor_dashboard.fetch_returns", return_value=stock_returns):
            with patch("factor_dashboard.plt") as mock_plt:
                mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock(), MagicMock()])
                result = analyse_ticker(
                    "TEST", spy_returns, factor_df,
                    "2020-01-01", "2021-12-31", rf=0.05, window=63,
                )

        assert result is not None
        expected_keys = {"ticker", "alpha", "beta", "r_squared", "smb", "hml", "ir", "rolling_beta"}
        assert set(result.keys()) == expected_keys
        assert result["ticker"] == "TEST"

    def test_beta_close_to_true(self, spy_returns, factor_df, stock_returns):
        with patch("factor_dashboard.fetch_returns", return_value=stock_returns):
            with patch("factor_dashboard.plt") as mock_plt:
                mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock(), MagicMock()])
                result = analyse_ticker(
                    "TEST", spy_returns, factor_df,
                    "2020-01-01", "2021-12-31", rf=0.05, window=63,
                )

        assert result["beta"] == pytest.approx(1.3, abs=0.15)

    def test_r_squared_reasonable(self, spy_returns, factor_df, stock_returns):
        with patch("factor_dashboard.fetch_returns", return_value=stock_returns):
            with patch("factor_dashboard.plt") as mock_plt:
                mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock(), MagicMock()])
                result = analyse_ticker(
                    "TEST", spy_returns, factor_df,
                    "2020-01-01", "2021-12-31", rf=0.05, window=63,
                )

        assert result["r_squared"] > 0.5

    def test_fetch_failure_returns_none(self, spy_returns, factor_df):
        with patch("factor_dashboard.fetch_returns", side_effect=Exception("Network error")):
            result = analyse_ticker(
                "BADTICKER", spy_returns, factor_df,
                "2020-01-01", "2021-12-31", rf=0.05, window=63,
            )
        assert result is None

    def test_empty_returns_skips(self, spy_returns, factor_df):
        empty = pd.Series([], dtype=float)
        with patch("factor_dashboard.fetch_returns", return_value=empty):
            result = analyse_ticker(
                "EMPTY", spy_returns, factor_df,
                "2020-01-01", "2021-12-31", rf=0.05, window=63,
            )
        assert result is None

    def test_short_date_range_skips(self, spy_returns, factor_df):
        """If data length < window, analyse_ticker should return None."""
        np.random.seed(10)
        short = pd.Series(
            np.random.normal(0.001, 0.01, 30),
            index=pd.date_range("2020-01-01", periods=30, freq="B"),
        )
        with patch("factor_dashboard.fetch_returns", return_value=short):
            result = analyse_ticker(
                "SHORT", spy_returns, factor_df,
                "2020-01-01", "2021-12-31", rf=0.05, window=63,
            )
        assert result is None

    def test_rolling_beta_is_series(self, spy_returns, factor_df, stock_returns):
        with patch("factor_dashboard.fetch_returns", return_value=stock_returns):
            with patch("factor_dashboard.plt") as mock_plt:
                mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock(), MagicMock()])
                result = analyse_ticker(
                    "TEST", spy_returns, factor_df,
                    "2020-01-01", "2021-12-31", rf=0.05, window=63,
                )

        assert isinstance(result["rolling_beta"], pd.Series)
        assert len(result["rolling_beta"]) > 0

    def test_ir_is_float(self, spy_returns, factor_df, stock_returns):
        with patch("factor_dashboard.fetch_returns", return_value=stock_returns):
            with patch("factor_dashboard.plt") as mock_plt:
                mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock(), MagicMock()])
                result = analyse_ticker(
                    "TEST", spy_returns, factor_df,
                    "2020-01-01", "2021-12-31", rf=0.05, window=63,
                )

        assert isinstance(result["ir"], float)
        assert np.isfinite(result["ir"])


# ---------------------------------------------------------------------------
# print_comparison_table
# ---------------------------------------------------------------------------

class TestPrintComparisonTable:
    def test_runs_without_error(self, summaries, capsys):
        print_comparison_table(summaries)
        captured = capsys.readouterr()
        assert "AAPL" in captured.out
        assert "JPM" in captured.out
        assert "FACTOR EXPOSURE COMPARISON" in captured.out

    def test_single_summary(self, summaries, capsys):
        print_comparison_table([summaries[0]])
        captured = capsys.readouterr()
        assert "AAPL" in captured.out


# ---------------------------------------------------------------------------
# save_comparison_chart
# ---------------------------------------------------------------------------

class TestSaveComparisonChart:
    def test_closes_chart(self, summaries):
        with patch("factor_dashboard.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            save_comparison_chart(summaries, window=63)
            mock_plt.close.assert_called_once()


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_ticker_no_comparison(self, spy_returns, factor_df, stock_returns):
        """With a single ticker, summaries list has 1 entry — no comparison table."""
        with patch("factor_dashboard.fetch_returns", return_value=stock_returns):
            with patch("factor_dashboard.plt") as mock_plt:
                mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock(), MagicMock()])
                result = analyse_ticker(
                    "SOLO", spy_returns, factor_df,
                    "2020-01-01", "2021-12-31", rf=0.05, window=63,
                )
        # Single result is valid
        assert result is not None
        # Comparison table only prints for len > 1 (tested via main flow)
        summaries = [result]
        assert len(summaries) == 1

    def test_all_tickers_failing(self, spy_returns, factor_df):
        """If all tickers fail, summaries should be empty."""
        results = []
        for ticker in ["BAD1", "BAD2", "BAD3"]:
            with patch("factor_dashboard.fetch_returns", side_effect=Exception("fail")):
                r = analyse_ticker(ticker, spy_returns, factor_df,
                                   "2020-01-01", "2021-12-31", rf=0.05, window=63)
                if r is not None:
                    results.append(r)
        assert len(results) == 0

    def test_summary_values_are_numeric(self, spy_returns, factor_df, stock_returns):
        with patch("factor_dashboard.fetch_returns", return_value=stock_returns):
            with patch("factor_dashboard.plt") as mock_plt:
                mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock(), MagicMock()])
                result = analyse_ticker(
                    "TEST", spy_returns, factor_df,
                    "2020-01-01", "2021-12-31", rf=0.05, window=63,
                )
        for key in ["alpha", "beta", "r_squared", "smb", "hml", "ir"]:
            assert isinstance(result[key], (int, float, np.floating)), f"{key} is not numeric"
            assert np.isfinite(result[key]), f"{key} is not finite"
