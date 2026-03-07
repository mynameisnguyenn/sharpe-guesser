"""
Models: Four Volatility Forecasting Approaches
================================================

This module implements four volatility forecasting models, ordered from simplest
to most sophisticated:

1. **EWMA (RiskMetrics)** — exponentially weighted moving average. The industry
   standard baseline. Lambda=0.94 gives a ~60-day half-life, which is what
   JPMorgan's RiskMetrics system used. No fitting required.

2. **GARCH(1,1)** — the workhorse of academic vol modeling since Bollerslev (1986).
   Captures volatility clustering: big moves follow big moves. Uses an expanding
   window with periodic refitting to avoid lookahead bias.

3. **HAR-RV (Corsi 2009)** — Heterogeneous Autoregressive model of Realized
   Volatility. Regresses forward RV on daily, weekly, and monthly RV. Captures
   the "cascade" structure: different market participants operate at different
   horizons (day-traders, swing traders, macro investors).

4. **VIX Implied** — the market's own forecast. VIX is the risk-neutral
   expectation of 30-day S&P 500 vol. No model to fit — just alignment. This
   is the benchmark that model-based forecasts need to beat.

The key function is `run_all_forecasts`, which runs all four models and returns
a single DataFrame with aligned predictions and realized (actual) vol for
evaluation.

All volatilities are returned ANNUALIZED in decimal form (e.g., 0.20 = 20% vol).
"""

import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. EWMA (RiskMetrics)
# ---------------------------------------------------------------------------
# The simplest model: a weighted average of past squared returns where recent
# observations get exponentially more weight. Lambda=0.94 is the RiskMetrics
# standard for daily data — it corresponds to roughly a 60-day half-life.
#
# On a risk desk, this is the default vol estimate. It's what you'd see in
# a basic VaR model. It reacts to recent shocks faster than a simple moving
# average, but it has no "mean reversion" — unlike GARCH, it doesn't pull
# vol back toward a long-run level after a spike.

def ewma_vol(
    returns: pd.Series,
    lam: float = 0.94,
    min_periods: int = 22,
) -> pd.Series:
    """Exponentially weighted moving average volatility (RiskMetrics)."""
    squared_returns = returns ** 2
    # com = lambda / (1 - lambda) maps the decay factor to pandas ewm convention
    ewma_var = squared_returns.ewm(com=lam / (1 - lam), min_periods=min_periods).mean()
    ewma_sigma = np.sqrt(ewma_var * 252)
    ewma_sigma.name = "ewma"
    return ewma_sigma


# ---------------------------------------------------------------------------
# 2. GARCH(1,1)
# ---------------------------------------------------------------------------
# sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}
#
# GARCH captures two things EWMA misses:
#   1. Mean reversion: vol reverts to omega/(1 - alpha - beta) over time
#   2. Fitted parameters: alpha and beta are estimated from data, not fixed
#
# The expanding window approach prevents lookahead bias. We refit every
# `refit_every` days to keep runtime reasonable — fitting GARCH on 10+ years
# of daily data for every single day would be extremely slow. Between refits,
# we use the recursive formula with the last fitted parameters.
#
# The arch package convention: pass returns in PERCENTAGE form (returns * 100).
# Conditional variance comes back in pct^2, so divide by 10000 to get decimal.

def garch_vol(
    returns: pd.Series,
    refit_every: int = 22,
    min_history: int = 252,
) -> pd.Series:
    """GARCH(1,1) volatility forecast with expanding window."""
    from arch import arch_model

    n = len(returns)
    result = pd.Series(np.nan, index=returns.index, name="garch")

    if n < min_history:
        return result

    # Scale to percentage (arch convention)
    pct_returns = returns * 100

    # Track fitted parameters for recursive updates between refits
    omega, alpha, beta = None, None, None
    last_var_pct = None  # last conditional variance in pct^2
    last_fit_idx = None

    for t in range(min_history, n):
        need_refit = (
            omega is None
            or (t - last_fit_idx) >= refit_every
        )

        if need_refit:
            train = pct_returns.iloc[:t]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    am = arch_model(
                        train,
                        vol="Garch",
                        p=1,
                        q=1,
                        mean="Zero",
                        rescale=False,
                    )
                    res = am.fit(disp="off", show_warning=False)
                    omega = res.params.get("omega", None)
                    alpha = res.params.get("alpha[1]", None)
                    beta = res.params.get("beta[1]", None)

                    if omega is None or alpha is None or beta is None:
                        continue

                    # Last in-sample conditional variance (pct^2)
                    last_var_pct = res.conditional_volatility.iloc[-1] ** 2
                    last_fit_idx = t
                except Exception:
                    # Fitting can fail on degenerate data — skip this day
                    continue

        if omega is None:
            continue

        # Recursive 1-step forecast: sigma^2_t = omega + alpha*r^2_{t-1} + beta*sigma^2_{t-1}
        r_prev_pct = pct_returns.iloc[t - 1]
        var_pct = omega + alpha * r_prev_pct ** 2 + beta * last_var_pct

        # Convert: pct^2 → decimal variance → annualized vol
        result.iloc[t] = np.sqrt(var_pct / 10000 * 252)

        last_var_pct = var_pct

    return result


# ---------------------------------------------------------------------------
# 3. HAR-RV (Corsi 2009)
# ---------------------------------------------------------------------------
# The HAR model is one of the most successful vol forecasting models in
# academic finance. The idea: realized vol at different horizons (daily,
# weekly, monthly) captures the activity of different types of traders.
# Day-traders drive daily RV, swing traders drive weekly RV, macro/pension
# funds drive monthly RV.
#
# The regression:
#   RV_forward_22d = a + b1*RV_1d + b2*RV_5d + b3*RV_22d + epsilon
#
# We use an expanding-window OLS to avoid lookahead bias, refitting every
# `refit_every` days. Between refits, we apply the last fitted coefficients
# to today's features.

def har_rv_vol(
    returns: pd.Series,
    min_history: int = 252,
    refit_every: int = 22,
) -> pd.Series:
    """Heterogeneous Autoregressive model of Realized Volatility (Corsi 2009)."""
    import statsmodels.api as sm
    from .realized_vol import har_features, forward_rv

    # Build features and target
    features = har_features(returns)        # columns: rv_d, rv_w, rv_m
    target = forward_rv(returns, window=22) # 22-day forward realized vol

    # Align on common index
    combined = features.join(target.rename("target"), how="inner").dropna()

    if len(combined) < min_history:
        return pd.Series(np.nan, index=returns.index, name="har")

    feature_cols = ["rv_d", "rv_w", "rv_m"]
    result = pd.Series(np.nan, index=returns.index, name="har")

    last_params = None
    last_fit_idx = None

    for t in range(min_history, len(combined)):
        need_refit = (
            last_params is None
            or (t - last_fit_idx) >= refit_every
        )

        if need_refit:
            train = combined.iloc[:t]
            X_train = sm.add_constant(train[feature_cols])
            y_train = train["target"]

            try:
                ols_res = sm.OLS(y_train, X_train).fit()
                last_params = ols_res.params
                last_fit_idx = t
            except Exception:
                continue

        if last_params is None:
            continue

        # Predict at time t using current features
        row = combined.iloc[t]
        x_t = np.array([1.0, row["rv_d"], row["rv_w"], row["rv_m"]])
        pred = x_t @ last_params

        # Vol can't be negative
        pred = max(pred, 0.0)

        # Map back to original index
        orig_idx = combined.index[t]
        if orig_idx in result.index:
            result.loc[orig_idx] = pred

    return result


# ---------------------------------------------------------------------------
# 4. VIX IMPLIED
# ---------------------------------------------------------------------------
# VIX is the CBOE Volatility Index — the market's risk-neutral expectation of
# 30-day S&P 500 vol. It's quoted as annualized percentage (e.g., VIX = 20
# means the market expects ~20% annualized vol over the next 30 days).
#
# No model to fit — just convert from percentage to decimal. This is the
# benchmark: if your fancy GARCH or HAR model can't beat VIX, the market
# already knows everything your model knows.

def vix_implied_vol(vix: pd.Series) -> pd.Series:
    """Convert VIX index to annualized decimal vol forecast."""
    result = vix / 100
    result.name = "vix"
    return result


# ---------------------------------------------------------------------------
# 5. FORECAST RUNNER
# ---------------------------------------------------------------------------
# Runs all four models and returns a combined DataFrame with aligned forecasts
# and the realized (actual) forward vol for evaluation.

def run_all_forecasts(returns: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Run all 4 vol forecast models and return aligned DataFrame."""
    from .realized_vol import forward_rv

    # Compute the target: 22-day forward realized vol
    realized = forward_rv(returns, window=22)
    realized.name = "realized"

    # Run each model
    ewma = ewma_vol(returns)
    garch = garch_vol(returns)
    har = har_rv_vol(returns)
    vix_forecast = vix_implied_vol(vix)

    # Combine into one DataFrame
    df = pd.DataFrame({
        "realized": realized,
        "ewma": ewma,
        "garch": garch,
        "har": har,
        "vix": vix_forecast,
    })

    # Keep only rows where all columns are available
    df = df.dropna()

    return df
