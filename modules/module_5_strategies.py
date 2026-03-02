"""
Module 5: Systematic Strategies & Backtesting
===============================================

This is where it all comes together. You'll build simple systematic
strategies, backtest them properly, and evaluate them using everything
from the previous modules.

This module covers the two most fundamental quant strategies:
    1. Momentum — "winners keep winning"
    2. Mean reversion — "what goes up must come down"

And the backtesting framework to evaluate them honestly.

Topics:
    1. Signal construction
    2. Simple momentum strategy
    3. Simple mean reversion (pairs trading)
    4. Backtesting engine
    5. Strategy evaluation & the danger of overfitting

Run:
    python -m modules.module_5_strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# BACKTESTING ENGINE
# ---------------------------------------------------------------------------
# A simple but honest backtester. The key principles:
#   - No lookahead bias (you can only use data available at the time)
#   - Include transaction costs
#   - Report realistic metrics
#
# This is intentionally simple. Production systems use event-driven
# architectures, but for learning, a vectorised approach is clearer.

class SimpleBacktester:
    """
    Vectorised backtester for long/short strategies.

    Takes a signal (pd.Series of position weights) and returns
    (pd.Series of daily returns) and computes strategy performance.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        positions: pd.DataFrame,
        transaction_cost_bps: float = 10,  # 10 bps per trade
        name: str = "Strategy",
    ):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns for each asset.
        positions : pd.DataFrame
            Target position weights for each asset (same columns as returns).
            Positions are established at close, returns earned next day.
        transaction_cost_bps : float
            Round-trip transaction cost in basis points.
        """
        self.name = name
        self.returns = returns
        self.positions = positions
        self.tc_bps = transaction_cost_bps

        # Align
        common_idx = returns.index.intersection(positions.index)
        self.returns = returns.loc[common_idx]
        self.positions = positions.loc[common_idx]

        # Strategy returns: position at t * return at t+1
        self.strategy_returns = (self.positions.shift(1) * self.returns).sum(axis=1).dropna()

        # Transaction costs: cost proportional to position changes
        turnover = self.positions.diff().abs().sum(axis=1)
        tc = turnover * (transaction_cost_bps / 10_000)
        self.strategy_returns_net = self.strategy_returns - tc.reindex(self.strategy_returns.index, fill_value=0)

        # Cumulative
        self.cumulative = (1 + self.strategy_returns_net).cumprod()

    def stats(self) -> dict:
        """Compute strategy statistics."""
        rets = self.strategy_returns_net
        cum = self.cumulative

        ann_ret = rets.mean() * TRADING_DAYS
        ann_vol = rets.std() * np.sqrt(TRADING_DAYS)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        # Drawdown
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = dd.min()

        # Sortino
        downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS)
        sortino = ann_ret / downside if downside > 0 else 0

        # Calmar
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # Win rate
        win_rate = (rets > 0).mean()

        # Turnover
        turnover = self.positions.diff().abs().sum(axis=1).mean() * TRADING_DAYS

        return {
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "annual_turnover": turnover,
            "total_return": cum.iloc[-1] - 1,
        }

    def print_stats(self):
        """Pretty-print strategy statistics."""
        s = self.stats()
        print(f"\n  {self.name} — Performance Summary")
        print(f"  {'-'*45}")
        print(f"    Annual return   : {s['annual_return']:>8.2%}")
        print(f"    Annual vol      : {s['annual_vol']:>8.2%}")
        print(f"    Sharpe ratio    : {s['sharpe']:>8.2f}")
        print(f"    Sortino ratio   : {s['sortino']:>8.2f}")
        print(f"    Calmar ratio    : {s['calmar']:>8.2f}")
        print(f"    Max drawdown    : {s['max_drawdown']:>8.2%}")
        print(f"    Win rate        : {s['win_rate']:>8.1%}")
        print(f"    Annual turnover : {s['annual_turnover']:>8.1f}x")
        print(f"    Total return    : {s['total_return']:>8.2%}")

    def plot(self):
        """Plot cumulative return and drawdown."""
        cum = self.cumulative
        peak = cum.cummax()
        dd = (cum - peak) / peak

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

        ax1.plot(cum, linewidth=1, color="steelblue")
        ax1.set_title(f"{self.name} — Cumulative Return")
        ax1.set_ylabel("Growth of $1")
        ax1.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"module5_{self.name.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=120)
        plt.close()
        print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# 1. MOMENTUM STRATEGY
# ---------------------------------------------------------------------------
# The idea: stocks that have gone up recently tend to keep going up.
# Stocks that have gone down tend to keep going down. This is the most
# well-documented anomaly in finance.
#
# How it works:
#   1. Rank stocks by their trailing return (e.g., past 12 months)
#   2. Go long the top quintile, short the bottom quintile
#   3. Rebalance monthly

def momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 252,    # 12-month momentum
    skip: int = 21,         # skip most recent month (short-term reversal)
) -> pd.DataFrame:
    """
    Compute cross-sectional momentum signal.

    For each date, rank stocks by their trailing return (excluding
    the most recent month to avoid short-term reversal).
    """
    # Trailing return, skipping the most recent 'skip' days
    trailing_ret = prices.shift(skip).pct_change(lookback - skip)

    # Rank into z-scores (cross-sectionally)
    ranked = trailing_ret.rank(axis=1, pct=True)

    # Convert to positions: long top, short bottom, zero middle
    # Demean so it's dollar-neutral (long and short offset)
    positions = ranked.sub(ranked.mean(axis=1), axis=0)

    # Normalise so absolute weights sum to 1
    abs_sum = positions.abs().sum(axis=1)
    positions = positions.div(abs_sum, axis=0)

    return positions


def run_momentum(tickers: list, start: str, end: str) -> SimpleBacktester:
    """Run a simple momentum strategy."""
    prices = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    returns = prices.pct_change().dropna()

    positions = momentum_signal(prices)

    bt = SimpleBacktester(
        returns=returns,
        positions=positions,
        transaction_cost_bps=10,
        name="Cross-Sectional Momentum",
    )
    return bt


# ---------------------------------------------------------------------------
# 2. MEAN REVERSION (PAIRS TRADING)
# ---------------------------------------------------------------------------
# The opposite bet: when two correlated assets diverge, bet on convergence.
#
# Classic example: Coca-Cola vs Pepsi. They move together most of the time.
# When the spread widens, you go long the laggard and short the leader.
#
# The key question: is the spread stationary? If not, it might diverge
# forever and you'll bleed.

def compute_spread(
    prices_a: pd.Series,
    prices_b: pd.Series,
) -> tuple:
    """
    Compute the hedge ratio and spread between two assets using OLS.

    spread = prices_a - beta * prices_b
    """
    import statsmodels.api as sm

    # Regress A on B to get the hedge ratio
    X = sm.add_constant(prices_b)
    model = sm.OLS(prices_a, X).fit()
    beta = model.params.iloc[1]

    spread = prices_a - beta * prices_b
    return spread, beta


def zscore(series: pd.Series, window: int = 63) -> pd.Series:
    """Rolling z-score: how many standard deviations from the rolling mean."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ((series - mean) / std).dropna()


def pairs_trading_signal(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int = 63,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """
    Generate pairs trading positions.

    When z-score > entry_z: short A, long B (spread will narrow)
    When z-score < -entry_z: long A, short B (spread will widen back)
    Exit when |z-score| < exit_z
    """
    spread, beta = compute_spread(prices_a, prices_b)
    z = zscore(spread, window)

    # Position tracking
    pos_a = pd.Series(0.0, index=z.index)
    pos_b = pd.Series(0.0, index=z.index)

    in_trade = False
    current_side = 0  # +1 = long spread, -1 = short spread

    for i in range(1, len(z)):
        if not in_trade:
            if z.iloc[i] > entry_z:
                # Spread too high → short A, long B
                current_side = -1
                in_trade = True
            elif z.iloc[i] < -entry_z:
                # Spread too low → long A, short B
                current_side = 1
                in_trade = True
        else:
            if abs(z.iloc[i]) < exit_z:
                current_side = 0
                in_trade = False

        pos_a.iloc[i] = current_side * 0.5
        pos_b.iloc[i] = -current_side * 0.5 * beta

    positions = pd.DataFrame({
        prices_a.name or "A": pos_a,
        prices_b.name or "B": pos_b,
    })
    return positions, z


def run_pairs_trading(
    ticker_a: str,
    ticker_b: str,
    start: str,
    end: str,
) -> SimpleBacktester:
    """Run a simple pairs trading strategy."""
    prices = yf.download([ticker_a, ticker_b], start=start, end=end, progress=False)["Close"]
    returns = prices.pct_change().dropna()

    positions, z = pairs_trading_signal(
        prices[ticker_a],
        prices[ticker_b],
    )

    bt = SimpleBacktester(
        returns=returns,
        positions=positions,
        transaction_cost_bps=10,
        name=f"Pairs: {ticker_a}/{ticker_b}",
    )

    # Plot the spread z-score
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(z, linewidth=0.6, color="steelblue")
    ax1.axhline(1.5, color="red", linewidth=0.5, linestyle="--")
    ax1.axhline(-1.5, color="green", linewidth=0.5, linestyle="--")
    ax1.axhline(0, color="grey", linewidth=0.5)
    ax1.set_title(f"Spread Z-Score: {ticker_a} vs {ticker_b}")
    ax1.set_ylabel("Z-Score")

    ax2.plot(positions.iloc[:, 0], label=ticker_a, linewidth=0.6)
    ax2.plot(positions.iloc[:, 1], label=ticker_b, linewidth=0.6)
    ax2.set_title("Positions")
    ax2.set_ylabel("Weight")
    ax2.legend()

    plt.tight_layout()
    filename = f"module5_pairs_{ticker_a}_{ticker_b}_spread.png"
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"  Saved: {filename}")

    return bt


# ---------------------------------------------------------------------------
# 3. STRATEGY COMPARISON
# ---------------------------------------------------------------------------

def compare_strategies(strategies: list):
    """Plot multiple strategies on the same chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for bt in strategies:
        ax.plot(bt.cumulative, linewidth=1, label=f"{bt.name} (SR={bt.stats()['sharpe']:.2f})")

    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title("Strategy Comparison")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("module5_strategy_comparison.png", dpi=120)
    plt.close()
    print("  Saved: module5_strategy_comparison.png")


# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  MODULE 5: Systematic Strategies & Backtesting")
    print("=" * 60)

    start, end = "2018-01-01", "2025-12-31"

    # Momentum
    print("\n--- Cross-Sectional Momentum ---")
    mom_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM",
                   "XOM", "JNJ", "PG", "BAC", "CVX"]
    mom_bt = run_momentum(mom_tickers, start, end)
    mom_bt.print_stats()
    mom_bt.plot()

    # Pairs trading
    print("\n--- Pairs Trading ---")
    pairs_bt = run_pairs_trading("KO", "PEP", start, end)
    pairs_bt.print_stats()
    pairs_bt.plot()

    # Compare
    print("\n--- Strategy Comparison ---")
    compare_strategies([mom_bt, pairs_bt])

    print(f"\n{'='*60}")
    print("  KEY TAKEAWAYS:")
    print("=" * 60)
    print("  1. Momentum is the most robust anomaly in finance — it works across")
    print("     asset classes and time periods (but has sharp drawdowns)")
    print("  2. Mean reversion works when assets are cointegrated — test for")
    print("     stationarity before trading a spread (Module 1)")
    print("  3. Transaction costs matter enormously — a strategy with 2x annual")
    print("     turnover at 10bps costs 0.2% per year")
    print("  4. Sharpe alone isn't enough — look at drawdown, turnover, and")
    print("     whether the return comes from a few big bets or many small ones")
    print("  5. BEWARE OVERFITTING: if you test 100 parameter combinations,")
    print("     5 will look great by chance. Always hold out test data.")
    print()

    print("  EXERCISES:")
    print("  1. Change the momentum lookback from 12 months to 6 months and")
    print("     3 months. Which works better? (Be careful — that's data mining!)")
    print("  2. Find another pairs trade: try GS/MS or XOM/CVX. Test for")
    print("     cointegration first using the ADF test from Module 1.")
    print("  3. Add a volatility-targeting layer: scale positions so that the")
    print("     strategy targets 10% annual vol. How does the Sharpe change?")
    print("  4. Split the data in half. Optimise on the first half, test on the")
    print("     second half. Does the strategy survive out-of-sample?")
    print()

    print("  DANGER ZONES (things that will blow up a real strategy):")
    print("  - Survivorship bias: we only used stocks that still exist")
    print("  - Lookahead bias: make sure signals only use past data")
    print("  - Overfitting: more parameters = more risk of fooling yourself")
    print("  - Regime change: what worked 2010-2020 may not work 2020-2030")
    print("  - Liquidity: can you actually trade the size you're backtesting?")
    print()


if __name__ == "__main__":
    main()
