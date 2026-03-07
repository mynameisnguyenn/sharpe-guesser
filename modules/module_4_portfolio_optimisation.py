"""
Module 4: Portfolio Optimisation
=================================

This is where you go from "measuring risk" to "constructing portfolios."
This is the bridge module — once you can optimise a portfolio, you can
pitch ideas to PMs with real numbers behind them.

Topics:
    1. Mean-variance optimisation (Markowitz efficient frontier)
    2. Minimum variance portfolio
    3. Maximum Sharpe portfolio (tangency portfolio)
    4. Risk parity — equal risk contribution
    5. Black-Litterman — combine equilibrium with views

Run:
    python -m modules.module_4_portfolio_optimisation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def get_portfolio_data(tickers: list, start: str, end: str):
    """Fetch returns and compute expected returns / covariance."""
    prices = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    returns = prices.pct_change().dropna()

    # Annualised expected returns (historical mean — crude but standard)
    mu = returns.mean() * TRADING_DAYS
    # Annualised covariance matrix
    cov = returns.cov() * TRADING_DAYS

    return returns, mu, cov


# ---------------------------------------------------------------------------
# 1. PORTFOLIO MATH
# ---------------------------------------------------------------------------
# These are the building blocks. Every optimiser below uses these.

def portfolio_return(weights: np.ndarray, mu: pd.Series) -> float:
    """Expected annualised return of a portfolio."""
    return weights @ mu


def portfolio_volatility(weights: np.ndarray, cov: pd.DataFrame) -> float:
    """Annualised portfolio volatility (standard deviation)."""
    return np.sqrt(weights @ cov @ weights)


def portfolio_sharpe(
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.05,
) -> float:
    """Annualised Sharpe ratio of a portfolio."""
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    if vol < 1e-10:
        return 0.0
    return (ret - rf) / vol


# ---------------------------------------------------------------------------
# 2. EFFICIENT FRONTIER — MARKOWITZ (1952)
# ---------------------------------------------------------------------------
# The efficient frontier is the set of portfolios that give you the
# maximum return for each level of risk. Anything below it is suboptimal.
#
# Harry Markowitz won the Nobel Prize for this. It's elegant, but in
# practice it's sensitive to input estimates — small changes in expected
# returns produce wildly different portfolios. That's why quants developed
# the alternatives below.

def minimum_variance_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    allow_short: bool = False,
) -> dict:
    """Find the portfolio with the lowest possible volatility."""
    n = len(mu)
    w0 = np.ones(n) / n

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = None if allow_short else [(0, 1)] * n

    result = minimize(
        lambda w: portfolio_volatility(w, cov),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights = result.x
    return {
        "weights": weights,
        "return": portfolio_return(weights, mu),
        "volatility": portfolio_volatility(weights, cov),
        "sharpe": portfolio_sharpe(weights, mu, cov),
    }


def maximum_sharpe_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.05,
    allow_short: bool = False,
) -> dict:
    """
    Find the tangency portfolio — the one with the highest Sharpe ratio.

    This is the portfolio every rational investor should hold according
    to CAPM (combined with the risk-free asset).
    """
    n = len(mu)
    w0 = np.ones(n) / n

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = None if allow_short else [(0, 1)] * n

    # Minimise negative Sharpe (= maximise Sharpe)
    result = minimize(
        lambda w: -portfolio_sharpe(w, mu, cov, rf),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights = result.x
    return {
        "weights": weights,
        "return": portfolio_return(weights, mu),
        "volatility": portfolio_volatility(weights, cov),
        "sharpe": portfolio_sharpe(weights, mu, cov, rf),
    }


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 100,
    allow_short: bool = False,
) -> pd.DataFrame:
    """
    Trace out the efficient frontier by finding the minimum-variance
    portfolio for each target return level.
    """
    n = len(mu)
    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    results = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, mu) - t},
        ]
        bounds = None if allow_short else [(0, 1)] * n

        result = minimize(
            lambda w: portfolio_volatility(w, cov),
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            results.append({
                "return": portfolio_return(result.x, mu),
                "volatility": portfolio_volatility(result.x, cov),
                "sharpe": portfolio_sharpe(result.x, mu, cov),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 3. RISK PARITY
# ---------------------------------------------------------------------------
# Markowitz says "maximise Sharpe." Risk parity says "equalise risk
# contribution." Each asset contributes the same amount of risk to the
# portfolio.
#
# Why it's popular at hedge funds:
#   - No need to estimate expected returns (which are unreliable)
#   - More stable allocations over time
#   - Bridgewater's All Weather fund is the famous example

def risk_contribution(weights: np.ndarray, cov: pd.DataFrame) -> np.ndarray:
    """
    Compute each asset's marginal contribution to portfolio risk.

    Risk contribution of asset i = w_i * (Sigma @ w)_i / sigma_portfolio
    """
    port_vol = portfolio_volatility(weights, cov)
    marginal = cov.values @ weights
    return (weights * marginal) / port_vol


def risk_parity_portfolio(cov: pd.DataFrame) -> dict:
    """
    Find the risk parity portfolio where each asset contributes
    equal risk.
    """
    n = len(cov)
    w0 = np.ones(n) / n
    target_risk = 1.0 / n  # each asset contributes 1/n of total risk

    def objective(weights):
        rc = risk_contribution(weights, cov)
        rc_pct = rc / rc.sum()  # normalise to percentages
        # Minimise squared deviation from equal risk contribution
        return np.sum((rc_pct - target_risk) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 1)] * n  # no zeros, no shorts

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights = result.x
    rc = risk_contribution(weights, cov)
    rc_pct = rc / rc.sum()

    return {
        "weights": weights,
        "risk_contributions": rc_pct,
        "volatility": portfolio_volatility(weights, cov),
    }


# ---------------------------------------------------------------------------
# 4. BLACK-LITTERMAN
# ---------------------------------------------------------------------------
# The problem with Markowitz: expected returns are hard to estimate.
# Black-Litterman solves this by starting with the market equilibrium
# (what returns MUST be for the market portfolio to be optimal) and
# then tilting based on your views.
#
# This is how many multi-asset hedge funds actually allocate.

def implied_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: np.ndarray,
    risk_aversion: float = 2.5,
    rf: float = 0.05,
) -> pd.Series:
    """
    Reverse-optimise to find the implied returns that make the
    market portfolio optimal.

    Pi = delta * Sigma * w_market
    """
    pi = risk_aversion * cov @ market_weights
    return pi


def black_litterman(
    cov: pd.DataFrame,
    market_weights: np.ndarray,
    P: np.ndarray,       # pick matrix — which assets are in each view
    Q: np.ndarray,       # view returns — what you expect
    omega: np.ndarray,   # uncertainty in views
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> pd.Series:
    """
    Black-Litterman model: combine equilibrium returns with investor views.

    Parameters
    ----------
    P : np.ndarray, shape (k, n)
        Pick matrix — each row selects assets for a view.
        Example: P = [[1, 0, -1, 0]] means "asset 0 will outperform asset 2"
    Q : np.ndarray, shape (k,)
        View vector — expected excess return for each view.
    omega : np.ndarray, shape (k, k)
        Uncertainty matrix for views. Diagonal = independent views.
    """
    sigma = cov.values
    pi = implied_equilibrium_returns(cov, market_weights, risk_aversion).values

    # BL formula
    tau_sigma = tau * sigma
    inv_tau_sigma = np.linalg.inv(tau_sigma)
    inv_omega = np.linalg.inv(omega)

    # Posterior expected returns
    M1 = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
    M2 = inv_tau_sigma @ pi + P.T @ inv_omega @ Q

    bl_returns = pd.Series(M1 @ M2, index=cov.index)
    return bl_returns


# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    tickers: list,
    rf: float = 0.05,
):
    """Plot the efficient frontier with key portfolios marked."""
    ef = efficient_frontier(mu, cov)
    min_var = minimum_variance_portfolio(mu, cov)
    max_sr = maximum_sharpe_portfolio(mu, cov, rf)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Efficient frontier
    ax.plot(ef["volatility"], ef["return"], "b-", linewidth=2,
            label="Efficient Frontier")

    # Individual assets
    for i, ticker in enumerate(tickers):
        ax.scatter(
            np.sqrt(cov.iloc[i, i]),
            mu.iloc[i],
            s=80, zorder=5,
        )
        ax.annotate(ticker, (np.sqrt(cov.iloc[i, i]), mu.iloc[i]),
                     fontsize=9, ha="left", va="bottom")

    # Key portfolios
    ax.scatter(min_var["volatility"], min_var["return"],
               marker="*", s=300, c="green", zorder=5,
               label=f"Min Variance (SR={min_var['sharpe']:.2f})")
    ax.scatter(max_sr["volatility"], max_sr["return"],
               marker="*", s=300, c="red", zorder=5,
               label=f"Max Sharpe (SR={max_sr['sharpe']:.2f})")

    # Capital Market Line
    x_cml = np.linspace(0, ef["volatility"].max() * 1.2, 100)
    y_cml = rf + max_sr["sharpe"] * x_cml
    ax.plot(x_cml, y_cml, "r--", linewidth=0.8, alpha=0.5,
            label="Capital Market Line")

    ax.set_xlabel("Annualised Volatility")
    ax.set_ylabel("Annualised Return")
    ax.set_title("Efficient Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()


def print_portfolio(result: dict, tickers: list, name: str = ""):
    """Pretty-print portfolio weights and stats."""
    print(f"\n  {name}")
    print(f"  {'-'*45}")
    if "return" in result:
        print(f"    Expected return : {result['return']:>8.2%}")
    print(f"    Volatility      : {result['volatility']:>8.2%}")
    if "sharpe" in result:
        print(f"    Sharpe ratio    : {result['sharpe']:>8.2f}")
    print(f"    Weights:")
    for i, ticker in enumerate(tickers):
        w = result["weights"][i]
        bar = "#" * int(w * 40)
        print(f"      {ticker:<6} : {w:>6.1%}  {bar}")
    if "risk_contributions" in result:
        print(f"    Risk Contributions:")
        for i, ticker in enumerate(tickers):
            rc = result["risk_contributions"][i]
            print(f"      {ticker:<6} : {rc:>6.1%}")


# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  MODULE 4: Portfolio Optimisation")
    print("=" * 60)

    tickers = ["SPY", "TLT", "GLD", "QQQ", "XLE"]
    start, end = "2018-01-01", "2025-12-31"
    rf = 0.05

    returns, mu, cov = get_portfolio_data(tickers, start, end)

    # Equal weight baseline
    n = len(tickers)
    ew_weights = np.ones(n) / n
    ew = {
        "weights": ew_weights,
        "return": portfolio_return(ew_weights, mu),
        "volatility": portfolio_volatility(ew_weights, cov),
        "sharpe": portfolio_sharpe(ew_weights, mu, cov, rf),
    }
    print_portfolio(ew, tickers, "Equal Weight (Baseline)")

    # Minimum variance
    min_var = minimum_variance_portfolio(mu, cov)
    print_portfolio(min_var, tickers, "Minimum Variance")

    # Maximum Sharpe
    max_sr = maximum_sharpe_portfolio(mu, cov, rf)
    print_portfolio(max_sr, tickers, "Maximum Sharpe (Tangency)")

    # Risk parity
    rp = risk_parity_portfolio(cov)
    print_portfolio(rp, tickers, "Risk Parity")

    # Black-Litterman example: "QQQ will outperform XLE by 5%"
    print(f"\n  Black-Litterman Example")
    print(f"  View: QQQ will outperform XLE by 5% annually")

    market_weights = ew_weights  # simplified
    P = np.array([[0, 0, 0, 1, -1]])  # QQQ minus XLE
    Q = np.array([0.05])               # 5% outperformance
    omega = np.array([[0.001]])         # relatively confident

    bl_mu = black_litterman(cov, market_weights, P, Q, omega)
    print(f"\n    Equilibrium vs BL returns:")
    eq_mu = implied_equilibrium_returns(cov, market_weights)
    for ticker in tickers:
        eq_r = eq_mu[ticker] if ticker in eq_mu.index else 0
        bl_r = bl_mu[ticker] if ticker in bl_mu.index else 0
        print(f"      {ticker:<6} : Equilibrium {eq_r:>7.2%}  ->  BL {bl_r:>7.2%}")

    # Efficient frontier plot
    plot_efficient_frontier(mu, cov, tickers, rf)

    print(f"\n{'='*60}")
    print("  KEY TAKEAWAYS:")
    print("=" * 60)
    print("  1. Mean-variance is elegant but fragile — small input changes =")
    print("     big weight changes. Never use it blindly.")
    print("  2. Minimum variance doesn't need return estimates — more robust")
    print("  3. Risk parity equalises risk contribution — no return forecasts needed")
    print("  4. Black-Litterman starts from equilibrium and tilts with views —")
    print("     this is how institutional multi-asset funds actually allocate")
    print("  5. In practice, most funds add constraints (max position sizes,")
    print("     sector limits) to make Markowitz behave")
    print()

    print("  EXERCISES:")
    print("  1. Add a constraint: no single asset > 30%. How does it change the")
    print("     max Sharpe portfolio?")
    print("  2. Run risk parity with and without bonds (TLT). What happens to")
    print("     the equity allocation?")
    print("  3. In Black-Litterman, increase omega (less confident). How do the")
    print("     posterior returns change?")
    print("  4. Backtest: rebalance monthly to the max Sharpe portfolio. Compare")
    print("     to equal weight. Which has a higher realised Sharpe?")
    print()


if __name__ == "__main__":
    main()
