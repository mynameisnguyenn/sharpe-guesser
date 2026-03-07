"""
Risk Dashboard — Interactive Portfolio Risk Analysis
=====================================================

A self-contained Streamlit dashboard that computes risk metrics,
factor exposures, and comparative analysis for user-selected tickers.

Run: streamlit run app.py
"""

import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
import yfinance as yf
from scipy import stats

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Risk Dashboard", layout="wide")

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Data fetching (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers."""
    if len(tickers) == 1:
        df = yf.download(tickers[0], start=start, end=end, progress=False)["Close"]
        return pd.DataFrame(df).rename(columns={"Close": tickers[0]})
    df = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    return df


@st.cache_data(show_spinner=False)
def fetch_factor_etfs(start: str, end: str) -> pd.DataFrame:
    """Download prices for factor proxy ETFs: SPY, IWM, IWD, IWF."""
    tickers = ["SPY", "IWM", "IWD", "IWF"]
    prices = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    return prices


# ---------------------------------------------------------------------------
# Risk metric helpers
# ---------------------------------------------------------------------------

def compute_sharpe(daily_returns: pd.Series, annual_rf: float) -> float:
    daily_rf = annual_rf / TRADING_DAYS
    excess = daily_returns - daily_rf
    ann_excess = excess.mean() * TRADING_DAYS
    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
    return ann_excess / ann_vol if ann_vol > 0 else 0.0


def compute_sortino(daily_returns: pd.Series, annual_rf: float) -> float:
    daily_rf = annual_rf / TRADING_DAYS
    excess = daily_returns - daily_rf
    ann_excess = excess.mean() * TRADING_DAYS
    downside = excess[excess < 0]
    ann_dd = downside.std() * np.sqrt(TRADING_DAYS)
    return ann_excess / ann_dd if ann_dd > 0 and not np.isnan(ann_dd) else 0.0


def compute_calmar(prices: pd.Series, daily_returns: pd.Series, annual_rf: float) -> float:
    daily_rf = annual_rf / TRADING_DAYS
    ann_excess = (daily_returns.mean() - daily_rf) * TRADING_DAYS
    mdd = abs(compute_max_drawdown(prices))
    return ann_excess / mdd if mdd > 0 else 0.0


def compute_max_drawdown(prices: pd.Series) -> float:
    cummax = prices.cummax()
    dd = (prices - cummax) / cummax
    return dd.min()


def compute_drawdown_series(prices: pd.Series) -> pd.Series:
    cummax = prices.cummax()
    return (prices - cummax) / cummax


def var_historical(daily_returns: pd.Series, confidence: float) -> float:
    return daily_returns.quantile(1 - confidence)


def var_parametric(daily_returns: pd.Series, confidence: float) -> float:
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    return stats.norm.ppf(1 - confidence, mu, sigma)


def compute_cvar(daily_returns: pd.Series, confidence: float) -> float:
    var = var_historical(daily_returns, confidence)
    tail = daily_returns[daily_returns <= var]
    return tail.mean() if len(tail) > 0 else var


# ---------------------------------------------------------------------------
# Factor model helpers
# ---------------------------------------------------------------------------

def run_capm(stock_returns: pd.Series, market_returns: pd.Series, annual_rf: float) -> dict:
    daily_rf = annual_rf / TRADING_DAYS
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ["stock", "market"]
    y = aligned["stock"] - daily_rf
    x = sm.add_constant(aligned["market"] - daily_rf)
    model = sm.OLS(y, x).fit()
    return {
        "alpha_annual": model.params.iloc[0] * TRADING_DAYS,
        "beta": model.params.iloc[1],
        "r_squared": model.rsquared,
        "alpha_tstat": model.tvalues.iloc[0],
        "beta_tstat": model.tvalues.iloc[1],
    }


def run_ff3(stock_returns: pd.Series, factors: pd.DataFrame, annual_rf: float) -> dict:
    daily_rf = annual_rf / TRADING_DAYS
    aligned = pd.concat([stock_returns, factors], axis=1).dropna()
    y = aligned.iloc[:, 0] - daily_rf
    x = sm.add_constant(aligned.iloc[:, 1:])
    model = sm.OLS(y, x).fit()
    result = {
        "alpha_annual": model.params.iloc[0] * TRADING_DAYS,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "resid": model.resid,
    }
    for i, name in enumerate(factors.columns):
        result[f"beta_{name}"] = model.params.iloc[i + 1]
        result[f"tstat_{name}"] = model.tvalues.iloc[i + 1]
    return result


def compute_rolling_beta(stock_returns: pd.Series, market_returns: pd.Series, window: int) -> pd.Series:
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ["stock", "market"]
    rolling_cov = aligned["stock"].rolling(window).cov(aligned["market"])
    rolling_var = aligned["market"].rolling(window).var()
    return (rolling_cov / rolling_var).dropna()


def compute_information_ratio(alpha_annual: float, residuals: pd.Series) -> float:
    te = residuals.std() * np.sqrt(TRADING_DAYS)
    return alpha_annual / te if te > 0 else 0.0


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Risk Dashboard")
st.sidebar.markdown("---")

ticker_input = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL, MSFT, JPM")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

col_start, col_end = st.sidebar.columns(2)
start_date = col_start.date_input("Start date", value=datetime.date(2020, 1, 1))
end_date = col_end.date_input("End date", value=datetime.date.today())

annual_rf = st.sidebar.slider("Risk-free rate (%)", 0.0, 10.0, 5.0, 0.25) / 100.0
rolling_window = st.sidebar.slider("Rolling window (days)", 21, 252, 63)
var_confidence = st.sidebar.radio("VaR confidence level", [0.95, 0.99], format_func=lambda x: f"{x:.0%}")

st.sidebar.markdown("---")
st.sidebar.caption("Data from Yahoo Finance. Factor proxies: MKT=SPY, SMB=IWM-SPY, HML=IWD-IWF.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

if not tickers:
    st.warning("Enter at least one ticker in the sidebar.")
    st.stop()

start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

with st.spinner("Downloading price data..."):
    try:
        prices = fetch_prices(tickers, start_str, end_str)
        # Flatten MultiIndex columns if present
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)
        prices = prices.dropna(how="all")
        if prices.empty:
            st.error("No data returned. Check tickers and date range.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

# Compute returns
returns = prices.pct_change().dropna()

# Identify valid tickers (have data)
valid_tickers = [t for t in tickers if t in prices.columns and prices[t].notna().sum() > 1]
if not valid_tickers:
    st.error("None of the entered tickers returned valid data.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_risk, tab_factor, tab_compare = st.tabs(["Risk Overview", "Factor Exposure", "Comparison"])

# ============================= TAB 1: RISK OVERVIEW ========================

with tab_risk:
    st.header("Risk Overview")

    for ticker in valid_tickers:
        st.subheader(ticker)
        r = returns[ticker].dropna()
        p = prices[ticker].dropna()

        # Ratios
        sharpe = compute_sharpe(r, annual_rf)
        sortino = compute_sortino(r, annual_rf)
        calmar = compute_calmar(p, r, annual_rf)

        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c2.metric("Sortino Ratio", f"{sortino:.2f}")
        c3.metric("Calmar Ratio", f"{calmar:.2f}")

        with st.expander("What do these ratios mean?"):
            st.markdown(
                "- **Sharpe**: Excess return per unit of total volatility. Higher is better.\n"
                "- **Sortino**: Like Sharpe, but only penalises downside volatility.\n"
                "- **Calmar**: Excess return per unit of maximum drawdown. Shows return relative to worst-case pain."
            )

        # VaR / CVaR table
        var_data = []
        for conf in [0.95, 0.99]:
            var_data.append({
                "Confidence": f"{conf:.0%}",
                "VaR (Historical)": f"{var_historical(r, conf):.2%}",
                "VaR (Parametric)": f"{var_parametric(r, conf):.2%}",
                "CVaR (Historical)": f"{compute_cvar(r, conf):.2%}",
            })
        st.table(pd.DataFrame(var_data).set_index("Confidence"))

        with st.expander("What are VaR and CVaR?"):
            st.markdown(
                "- **VaR (Value at Risk)**: The worst daily loss expected at the given confidence level.\n"
                "- **CVaR (Conditional VaR)**: The average loss on days worse than the VaR threshold. "
                "Also called Expected Shortfall — regulators prefer it because it captures tail severity."
            )

        # Drawdown chart
        dd = compute_drawdown_series(p)
        mdd = compute_max_drawdown(p)

        # Normalize prices to 100 for display
        norm_prices = p / p.iloc[0] * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=norm_prices.index, y=norm_prices.values,
            name="Cumulative Price (rebased 100)", line=dict(color="#1f77b4"),
        ))
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100,
            name="Drawdown (%)", fill="tozeroy",
            line=dict(color="rgba(220, 53, 69, 0.7)"),
            fillcolor="rgba(220, 53, 69, 0.15)",
            yaxis="y2",
        ))
        fig.update_layout(
            title=f"{ticker} — Price & Drawdown (Max DD: {mdd:.1%})",
            yaxis=dict(title="Rebased Price"),
            yaxis2=dict(title="Drawdown (%)", overlaying="y", side="right", showgrid=False),
            height=400, margin=dict(t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")


# ============================= TAB 2: FACTOR EXPOSURE =====================

with tab_factor:
    st.header("Factor Exposure")

    with st.spinner("Downloading factor proxy ETFs..."):
        try:
            factor_prices = fetch_factor_etfs(start_str, end_str)
            if isinstance(factor_prices.columns, pd.MultiIndex):
                factor_prices.columns = factor_prices.columns.get_level_values(0)
            factor_rets = factor_prices.pct_change().dropna()
            spy_returns = factor_rets["SPY"]
            factors = pd.DataFrame({
                "MKT": factor_rets["SPY"],
                "SMB": factor_rets["IWM"] - factor_rets["SPY"],
                "HML": factor_rets["IWD"] - factor_rets["IWF"],
            })
            factor_data_ok = True
        except Exception as e:
            st.error(f"Could not download factor ETF data: {e}")
            factor_data_ok = False

    if factor_data_ok:
        # CAPM results table
        st.subheader("CAPM Regression Results")
        capm_rows = []
        for ticker in valid_tickers:
            r = returns[ticker].dropna()
            result = run_capm(r, spy_returns, annual_rf)
            capm_rows.append({
                "Ticker": ticker,
                "Alpha (ann.)": f"{result['alpha_annual']:.2%}",
                "Beta": f"{result['beta']:.3f}",
                "R-squared": f"{result['r_squared']:.1%}",
                "Alpha t-stat": f"{result['alpha_tstat']:.2f}",
                "Beta t-stat": f"{result['beta_tstat']:.2f}",
            })
        st.table(pd.DataFrame(capm_rows).set_index("Ticker"))

        with st.expander("What is CAPM?"):
            st.markdown(
                "The Capital Asset Pricing Model decomposes a stock's excess return into market exposure (beta) "
                "and unexplained return (alpha). A beta of 1.2 means the stock moves ~1.2x the market. "
                "Statistically significant alpha (|t-stat| > 2) suggests returns beyond what market exposure explains."
            )

        # FF3 results
        st.subheader("Fama-French 3-Factor Loadings")
        ff3_results = {}
        ff3_rows = []
        for ticker in valid_tickers:
            r = returns[ticker].dropna()
            result = run_ff3(r, factors, annual_rf)
            ff3_results[ticker] = result
            ff3_rows.append({
                "Ticker": ticker,
                "Alpha (ann.)": f"{result['alpha_annual']:.2%}",
                "MKT": f"{result['beta_MKT']:.3f}",
                "SMB": f"{result['beta_SMB']:.3f}",
                "HML": f"{result['beta_HML']:.3f}",
                "R-squared": f"{result['r_squared']:.1%}",
                "Adj R-sq": f"{result['adj_r_squared']:.1%}",
            })
        st.table(pd.DataFrame(ff3_rows).set_index("Ticker"))

        # Factor loadings bar chart
        loading_data = []
        for ticker in valid_tickers:
            res = ff3_results[ticker]
            for factor in ["MKT", "SMB", "HML"]:
                loading_data.append({
                    "Ticker": ticker,
                    "Factor": factor,
                    "Loading": res[f"beta_{factor}"],
                })
        loading_df = pd.DataFrame(loading_data)
        fig_bar = px.bar(
            loading_df, x="Factor", y="Loading", color="Ticker",
            barmode="group", title="Factor Loadings by Ticker",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_bar.update_layout(height=400, margin=dict(t=40, b=30))
        st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander("What are Fama-French factors?"):
            st.markdown(
                "- **MKT**: Market excess return (beta to SPY).\n"
                "- **SMB (Small Minus Big)**: Small-cap premium. Positive loading = behaves like small caps.\n"
                "- **HML (High Minus Low)**: Value premium. Positive loading = behaves like value stocks.\n\n"
                "Factor proxies used: MKT=SPY, SMB=IWM-SPY, HML=IWD-IWF."
            )

        # Information ratio
        st.subheader("Information Ratio")
        ir_rows = []
        for ticker in valid_tickers:
            res = ff3_results[ticker]
            ir = compute_information_ratio(res["alpha_annual"], res["resid"])
            ir_rows.append({"Ticker": ticker, "Information Ratio": f"{ir:.2f}"})
        st.table(pd.DataFrame(ir_rows).set_index("Ticker"))

        with st.expander("What is the Information Ratio?"):
            st.markdown(
                "IR = annualised alpha / annualised tracking error. It measures alpha efficiency. "
                "IR > 0.5 is good; IR > 1.0 is exceptional."
            )

        # Rolling beta chart
        st.subheader(f"Rolling Beta ({rolling_window}-day window)")
        fig_rb = go.Figure()
        for ticker in valid_tickers:
            r = returns[ticker].dropna()
            rb = compute_rolling_beta(r, spy_returns, rolling_window)
            fig_rb.add_trace(go.Scatter(
                x=rb.index, y=rb.values, name=ticker, mode="lines",
            ))
        fig_rb.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Beta = 1")
        fig_rb.update_layout(
            title=f"Rolling Beta to SPY ({rolling_window}-day window)",
            yaxis_title="Beta", height=450, margin=dict(t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_rb, use_container_width=True)


# ============================= TAB 3: COMPARISON ==========================

with tab_compare:
    st.header("Comparison")

    if len(valid_tickers) < 2:
        st.info("Enter at least two tickers for comparison.")
    else:
        # Side-by-side metrics table
        st.subheader("Side-by-Side Metrics")
        comp_rows = []
        for ticker in valid_tickers:
            r = returns[ticker].dropna()
            p = prices[ticker].dropna()
            ann_return = r.mean() * TRADING_DAYS
            ann_vol = r.std() * np.sqrt(TRADING_DAYS)
            comp_rows.append({
                "Ticker": ticker,
                "Ann. Return": f"{ann_return:.2%}",
                "Ann. Volatility": f"{ann_vol:.2%}",
                "Sharpe": f"{compute_sharpe(r, annual_rf):.2f}",
                "Sortino": f"{compute_sortino(r, annual_rf):.2f}",
                "Calmar": f"{compute_calmar(p, r, annual_rf):.2f}",
                "Max Drawdown": f"{compute_max_drawdown(p):.2%}",
                f"VaR {var_confidence:.0%}": f"{var_historical(r, var_confidence):.2%}",
                f"CVaR {var_confidence:.0%}": f"{compute_cvar(r, var_confidence):.2%}",
            })
        st.table(pd.DataFrame(comp_rows).set_index("Ticker"))

        # Cumulative returns chart
        st.subheader("Cumulative Returns")
        cum_returns = (1 + returns[valid_tickers]).cumprod()
        fig_cum = go.Figure()
        for ticker in valid_tickers:
            fig_cum.add_trace(go.Scatter(
                x=cum_returns.index, y=cum_returns[ticker].values,
                name=ticker, mode="lines",
            ))
        fig_cum.update_layout(
            title="Cumulative Returns (Growth of $1)",
            yaxis_title="Growth of $1", height=450, margin=dict(t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # Correlation heatmap
        st.subheader("Return Correlation")
        corr = returns[valid_tickers].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, title="Daily Return Correlation Matrix",
            labels=dict(color="Correlation"),
        )
        fig_corr.update_layout(height=450, margin=dict(t=40, b=30))
        st.plotly_chart(fig_corr, use_container_width=True)
