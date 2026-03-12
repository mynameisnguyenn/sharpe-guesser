"""
Regime Detection Dashboard — Interactive Streamlit App
=======================================================

Self-contained dashboard (no imports from src/) that fits an HMM to SPY + VIX,
detects market regimes, and shows how factor behavior and risk metrics shift.

4 tabs:
    1. Regime Overview — current regime, price chart with regime shading, summary table
    2. Factor Behavior — factor stats by regime, correlation heatmaps
    3. Risk Metrics — unconditional vs regime-conditional VaR/CVaR/vol
    4. Transitions — transition matrix, expected durations, regime frequency by year

Run: streamlit run app.py
"""

import datetime
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
import yfinance as yf
from hmmlearn.hmm import GaussianHMM


st.set_page_config(page_title="Regime Detection Dashboard", layout="wide")

TRADING_DAYS = 252
REGIME_COLORS = {"Bull": "#2ecc71", "Stress": "#f39c12", "Crisis": "#e74c3c"}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_data(start: str, end: str) -> tuple[pd.Series, pd.Series]:
    """Download SPY prices and VIX."""
    spy_raw = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy = spy_raw["Adj Close"].rename("SPY")

    vix_raw = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    vix = vix_raw["Close"].rename("VIX")
    return spy, vix


@st.cache_data(show_spinner=False)
def fetch_factor_etfs(start: str, end: str) -> pd.DataFrame:
    """Download factor proxy ETFs: SPY, IWM, IWD, IWF."""
    tickers = ["SPY", "IWM", "IWD", "IWF"]
    prices = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    return prices


# ---------------------------------------------------------------------------
# HMM helpers (inlined for self-containment)
# ---------------------------------------------------------------------------

def build_features(returns: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Build HMM feature matrix: [return, vix_z, rv]."""
    combined = pd.DataFrame({"return": returns, "vix": vix}).dropna()
    vix_mean = combined["vix"].expanding(min_periods=63).mean()
    vix_std = combined["vix"].expanding(min_periods=63).std()
    combined["vix_z"] = (combined["vix"] - vix_mean) / vix_std
    combined["rv"] = combined["return"].rolling(22).std() * np.sqrt(TRADING_DAYS)
    return combined[["return", "vix_z", "rv"]].dropna()


@st.cache_data(show_spinner=False)
def fit_and_predict(
    _features_values: np.ndarray,
    _features_index: pd.DatetimeIndex,
    n_states: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fit HMM and return regimes, transition matrix, means, labels."""
    model = GaussianHMM(
        n_components=n_states, covariance_type="full",
        n_iter=200, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_features_values)

    regimes = model.predict(_features_values)

    # Sort states by mean return
    mean_returns = model.means_[:, 0]
    sorted_idx = np.argsort(mean_returns)
    if n_states == 2:
        names = ["Stress", "Bull"]
    else:
        names = ["Crisis", "Stress", "Bull"]

    label_map = {int(sorted_idx[i]): names[i] for i in range(n_states)}
    labeled = np.array([label_map[r] for r in regimes])

    return labeled, model.transmat_, model.means_, label_map


# ---------------------------------------------------------------------------
# Risk metric helpers
# ---------------------------------------------------------------------------

def var_hist(returns, confidence):
    return returns.quantile(1 - confidence)


def cvar_hist(returns, confidence):
    var = var_hist(returns, confidence)
    tail = returns[returns <= var]
    return tail.mean() if len(tail) > 0 else var


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Regime Detection")
st.sidebar.markdown("---")

col_s, col_e = st.sidebar.columns(2)
start_date = col_s.date_input("Start", value=datetime.date(2005, 1, 1))
end_date = col_e.date_input("End", value=datetime.date.today())

n_states = st.sidebar.radio("Number of regimes", [2, 3], index=1)
var_confidence = st.sidebar.radio("VaR confidence", [0.95, 0.99], format_func=lambda x: f"{x:.0%}")

st.sidebar.markdown("---")
st.sidebar.caption(
    "HMM fitted on SPY returns, VIX z-score, and realized vol. "
    "Factor proxies: MKT=SPY, SMB=IWM-SPY, HML=IWD-IWF."
)

# ---------------------------------------------------------------------------
# Load and process data
# ---------------------------------------------------------------------------

start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

with st.spinner("Downloading data and fitting HMM..."):
    try:
        spy_prices, vix = fetch_data(start_str, end_str)
    except Exception as e:
        st.error(f"Data download failed: {e}")
        st.stop()

    spy_returns = spy_prices.pct_change().dropna()
    features = build_features(spy_returns, vix)

    if len(features) < 252:
        st.error("Not enough data. Try a wider date range.")
        st.stop()

    labeled, transmat, means, label_map = fit_and_predict(
        features.values, features.index, n_states,
    )
    regimes = pd.Series(labeled, index=features.index, name="regime")

# ---------------------------------------------------------------------------
# Factor data
# ---------------------------------------------------------------------------

with st.spinner("Downloading factor proxy ETFs..."):
    try:
        factor_prices = fetch_factor_etfs(start_str, end_str)
        factor_rets = factor_prices.pct_change().dropna()
        factors = pd.DataFrame({
            "MKT": factor_rets["SPY"],
            "SMB": factor_rets["IWM"] - factor_rets["SPY"],
            "HML": factor_rets["IWD"] - factor_rets["IWF"],
        })
        factor_ok = True
    except Exception:
        factor_ok = False

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_factor, tab_risk, tab_trans = st.tabs([
    "Regime Overview", "Factor Behavior", "Risk Metrics", "Transitions",
])

# ========================= TAB 1: REGIME OVERVIEW =========================

with tab_overview:
    st.header("Regime Overview")

    # Current regime indicator
    current_regime = regimes.dropna().iloc[-1]
    regime_color = REGIME_COLORS.get(current_regime, "#999")

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Regime", current_regime)
    regime_counts = regimes.value_counts()
    c2.metric("Days Analyzed", f"{len(regimes):,}")
    c3.metric("Regime Since",
              regimes[regimes != current_regime].index[-1].strftime("%Y-%m-%d")
              if (regimes != current_regime).any() else "Start")

    # Price chart with regime shading
    st.subheader("SPY with Regime Shading")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spy_prices.index, y=spy_prices.values,
        name="SPY", line=dict(color="black", width=1),
    ))

    # Add colored rectangles for each regime block
    valid = regimes.dropna()
    if len(valid) > 0:
        prev = valid.iloc[0]
        start_dt = valid.index[0]
        for i in range(1, len(valid)):
            curr = valid.iloc[i]
            if curr != prev or i == len(valid) - 1:
                end_dt = valid.index[i]
                color = REGIME_COLORS.get(prev, "#ccc")
                fig.add_vrect(
                    x0=start_dt, x1=end_dt,
                    fillcolor=color, opacity=0.15,
                    layer="below", line_width=0,
                    annotation_text=prev if (end_dt - start_dt).days > 180 else "",
                    annotation_position="top left",
                    annotation_font_size=9,
                )
                prev = curr
                start_dt = end_dt

    fig.update_layout(
        height=500, margin=dict(t=30, b=30),
        yaxis_title="Price ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("Regime Summary")
    aligned = pd.DataFrame({
        "ret": spy_returns.reindex(regimes.index),
        "regime": regimes,
    }).dropna()

    summary_rows = []
    for regime in sorted(aligned["regime"].unique()):
        sub = aligned.loc[aligned["regime"] == regime, "ret"]
        n = len(sub)
        ann_ret = sub.mean() * TRADING_DAYS
        ann_vol = sub.std() * np.sqrt(TRADING_DAYS)
        sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
        summary_rows.append({
            "Regime": regime,
            "Days": n,
            "% of Total": f"{n / len(aligned) * 100:.1f}%",
            "Ann. Return": f"{ann_ret:.1%}",
            "Ann. Vol": f"{ann_vol:.1%}",
            "Sharpe": f"{sharpe:.2f}",
            f"VaR {var_confidence:.0%}": f"{var_hist(sub, var_confidence):.2%}",
            f"CVaR {var_confidence:.0%}": f"{cvar_hist(sub, var_confidence):.2%}",
        })
    st.table(pd.DataFrame(summary_rows).set_index("Regime"))


# ========================= TAB 2: FACTOR BEHAVIOR ========================

with tab_factor:
    st.header("Factor Behavior by Regime")

    if not factor_ok:
        st.warning("Could not download factor ETF data.")
    else:
        # Factor stats by regime
        st.subheader("Factor Statistics by Regime")
        factor_aligned = factors.join(regimes, how="inner").dropna(subset=["regime"])

        stats_rows = []
        for regime in sorted(factor_aligned["regime"].unique()):
            sub = factor_aligned.loc[factor_aligned["regime"] == regime]
            row = {"Regime": regime}
            for col in factors.columns:
                s = sub[col]
                ann_m = s.mean() * TRADING_DAYS
                ann_v = s.std() * np.sqrt(TRADING_DAYS)
                sh = ann_m / ann_v if ann_v > 1e-10 else 0.0
                row[f"{col} Mean"] = f"{ann_m:.1%}"
                row[f"{col} Vol"] = f"{ann_v:.1%}"
                row[f"{col} Sharpe"] = f"{sh:.2f}"
            stats_rows.append(row)
        st.table(pd.DataFrame(stats_rows).set_index("Regime"))

        # Factor Sharpe bar chart
        st.subheader("Factor Sharpe Ratio by Regime")
        bar_data = []
        for regime in sorted(factor_aligned["regime"].unique()):
            sub = factor_aligned.loc[factor_aligned["regime"] == regime]
            for col in factors.columns:
                s = sub[col]
                ann_m = s.mean() * TRADING_DAYS
                ann_v = s.std() * np.sqrt(TRADING_DAYS)
                sh = ann_m / ann_v if ann_v > 1e-10 else 0.0
                bar_data.append({"Regime": regime, "Factor": col, "Sharpe": sh})
        bar_df = pd.DataFrame(bar_data)
        fig_bar = px.bar(
            bar_df, x="Factor", y="Sharpe", color="Regime",
            barmode="group", title="Factor Sharpe Ratio by Market Regime",
            color_discrete_map=REGIME_COLORS,
        )
        fig_bar.update_layout(height=400, margin=dict(t=40, b=30))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Correlation heatmaps
        st.subheader("Factor Correlations by Regime")
        cols = st.columns(n_states)
        for i, regime in enumerate(sorted(factor_aligned["regime"].unique())):
            with cols[i]:
                sub = factor_aligned.loc[factor_aligned["regime"] == regime, factors.columns]
                corr = sub.corr()
                fig_hm = px.imshow(
                    corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title=f"{regime}",
                    labels=dict(color="Corr"),
                )
                fig_hm.update_layout(height=350, margin=dict(t=40, b=20))
                st.plotly_chart(fig_hm, use_container_width=True)


# ========================= TAB 3: RISK METRICS ===========================

with tab_risk:
    st.header("Risk Metrics: Unconditional vs Regime-Conditional")

    aligned_risk = pd.DataFrame({
        "ret": spy_returns.reindex(regimes.index),
        "regime": regimes,
    }).dropna()

    # VaR / CVaR comparison table
    st.subheader(f"VaR & CVaR at {var_confidence:.0%} Confidence")
    risk_rows = []

    # Unconditional
    all_ret = aligned_risk["ret"]
    risk_rows.append({
        "": "Unconditional",
        "VaR": f"{var_hist(all_ret, var_confidence):.2%}",
        "CVaR": f"{cvar_hist(all_ret, var_confidence):.2%}",
        "Ann. Vol": f"{all_ret.std() * np.sqrt(TRADING_DAYS):.1%}",
        "Days": len(all_ret),
    })

    for regime in sorted(aligned_risk["regime"].unique()):
        sub = aligned_risk.loc[aligned_risk["regime"] == regime, "ret"]
        risk_rows.append({
            "": regime,
            "VaR": f"{var_hist(sub, var_confidence):.2%}",
            "CVaR": f"{cvar_hist(sub, var_confidence):.2%}",
            "Ann. Vol": f"{sub.std() * np.sqrt(TRADING_DAYS):.1%}",
            "Days": len(sub),
        })
    st.table(pd.DataFrame(risk_rows).set_index(""))

    # Risk underestimation callout
    unc_var = var_hist(all_ret, var_confidence)
    crisis_label = "Crisis" if n_states == 3 else "Stress"
    if crisis_label in aligned_risk["regime"].unique():
        crisis_ret = aligned_risk.loc[aligned_risk["regime"] == crisis_label, "ret"]
        crisis_var = var_hist(crisis_ret, var_confidence)
        ratio = crisis_var / unc_var if abs(unc_var) > 1e-10 else 1.0
        st.info(
            f"**Risk underestimation:** {crisis_label}-regime {var_confidence:.0%} VaR "
            f"({crisis_var:.2%}) is {ratio:.1f}x the unconditional VaR ({unc_var:.2%}). "
            f"Standard risk metrics blend calm and crisis periods, understating tail risk."
        )

    # Vol bars by regime
    st.subheader("Annualized Volatility by Regime")
    vol_data = []
    for regime in sorted(aligned_risk["regime"].unique()):
        sub = aligned_risk.loc[aligned_risk["regime"] == regime, "ret"]
        vol_data.append({
            "Regime": regime,
            "Ann. Vol": sub.std() * np.sqrt(TRADING_DAYS),
        })
    vol_data.append({
        "Regime": "Unconditional",
        "Ann. Vol": all_ret.std() * np.sqrt(TRADING_DAYS),
    })
    vol_df = pd.DataFrame(vol_data)
    color_map = {**REGIME_COLORS, "Unconditional": "#3498db"}
    fig_vol = px.bar(
        vol_df, x="Regime", y="Ann. Vol", color="Regime",
        title="Annualized Volatility: Regime-Conditional vs Unconditional",
        color_discrete_map=color_map,
    )
    fig_vol.update_layout(height=400, margin=dict(t=40, b=30), showlegend=False)
    fig_vol.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_vol, use_container_width=True)

    # Return distribution overlay
    st.subheader("Return Distributions by Regime")
    fig_dist = go.Figure()
    for regime in sorted(aligned_risk["regime"].unique()):
        sub = aligned_risk.loc[aligned_risk["regime"] == regime, "ret"]
        fig_dist.add_trace(go.Histogram(
            x=sub, name=regime, opacity=0.5, nbinsx=80,
            marker_color=REGIME_COLORS.get(regime, "#999"),
            histnorm="probability density",
        ))
    fig_dist.update_layout(
        barmode="overlay", height=400, margin=dict(t=40, b=30),
        xaxis_title="Daily Return", yaxis_title="Density",
        title="Daily Return Distributions by Regime",
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ========================= TAB 4: TRANSITIONS ============================

with tab_trans:
    st.header("Regime Transitions")

    # Sort transition matrix labels
    if n_states == 2:
        order = ["Stress", "Bull"]
    else:
        order = ["Crisis", "Stress", "Bull"]

    sorted_idx = sorted(label_map.keys(), key=lambda k: order.index(label_map[k]))
    sorted_names = [label_map[k] for k in sorted_idx]
    tm = pd.DataFrame(
        transmat[np.ix_(sorted_idx, sorted_idx)],
        index=sorted_names, columns=sorted_names,
    )

    # Transition matrix heatmap
    st.subheader("Transition Probability Matrix")
    fig_tm = px.imshow(
        tm, text_auto=".3f", color_continuous_scale="YlOrRd",
        zmin=0, zmax=1, labels=dict(x="To", y="From", color="P"),
    )
    fig_tm.update_layout(height=400, margin=dict(t=30, b=30))
    st.plotly_chart(fig_tm, use_container_width=True)

    with st.expander("How to read this"):
        st.markdown(
            "Each cell shows the probability of transitioning from one regime (row) "
            "to another (column) on the next trading day. High diagonal values mean "
            "regimes are persistent — markets don't flip randomly between states."
        )

    # Expected durations
    st.subheader("Expected Regime Duration")
    dur_rows = []
    for state_idx, name in label_map.items():
        p_stay = transmat[state_idx, state_idx]
        dur_days = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else np.inf
        dur_rows.append({
            "Regime": name,
            "P(stay)": f"{p_stay:.3f}",
            "Expected Duration (days)": f"{dur_days:.0f}",
            "Expected Duration (months)": f"{dur_days / 21:.1f}",
        })
    st.table(pd.DataFrame(dur_rows).set_index("Regime"))

    # Regime frequency by year
    st.subheader("Regime Frequency by Year")
    regime_year = pd.DataFrame({"regime": regimes, "year": regimes.index.year}).dropna()
    freq = regime_year.groupby(["year", "regime"]).size().unstack(fill_value=0)
    # Normalize to proportions
    freq_pct = freq.div(freq.sum(axis=1), axis=0)

    fig_freq = go.Figure()
    for regime in order:
        if regime in freq_pct.columns:
            fig_freq.add_trace(go.Bar(
                x=freq_pct.index.astype(str),
                y=freq_pct[regime],
                name=regime,
                marker_color=REGIME_COLORS.get(regime, "#999"),
            ))
    fig_freq.update_layout(
        barmode="stack", height=400, margin=dict(t=40, b=30),
        title="Regime Proportion by Year",
        yaxis_title="Proportion", yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig_freq, use_container_width=True)
