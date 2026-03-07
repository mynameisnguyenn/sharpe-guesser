# Chapters 4-5: Multi-Factor Models & Understanding Factors

## Chapter 4: From One Factor to Many

### History
- **CAPM** (Sharpe, Lintner, Mossin, 1960s): single factor (market)
- **APT** (Ross, 1976): multiple factors — `r = alpha + beta_1*f_1 + ... + beta_m*f_m + epsilon`
- **Rosenberg & Marathe** (1976): use stock *characteristics* as factor loadings
- **Banz** (1981): small-cap stocks outperform large-cap — empirical challenge to CAPM
- **Fama & French** (1993): market + size + value (3-factor model)

### Three Approaches to Factor Models

| Approach | Data Needs | Performance | Interpretability |
|----------|-----------|-------------|-----------------|
| Time Series | Medium | Low | Medium-High |
| Fundamental (characteristic) | High | High | High |
| Statistical (PCA) | Low | High | Low |

**Fundamental models** are what practitioners actually use. Stock characteristics
(beta, industry, value ratios, etc.) are the loadings. Factor returns are *estimated*
via cross-sectional regression.

**Time series models** (e.g., regress stock returns on macro indicators) are more
common in academia.

**Statistical models** (PCA on return covariance) need minimal data but are hard
to interpret beyond factor 1 (≈ market).

### How a Fundamental Risk Model Works (daily process)
1. Receive stock characteristics → loadings matrix B (n stocks × m factors)
2. Cross-sectional regression: `r = Bf + epsilon` → estimates factor returns f
3. Residuals epsilon = idiosyncratic returns
4. Use time series of f and epsilon to estimate factor covariance and idio vols
5. Output: risk model (B, Omega_factor, sigma_idio)

### Factor-Mimicking Portfolios (FMPs)
- Byproduct of the regression — a portfolio whose returns track the factor return
- Used for performance attribution and hedging
- Industry FMP ≠ sector ETF (FMP has zero exposure to all other factors)

### Why the covariance matrix matters
- n=5000 stocks → 12.5M entries in covariance matrix
- Factor model reduces this: m factors (≈50) → manageable
- `Cov(returns) = B * Omega_factor * B' + Omega_idio`

---

## Chapter 5: Understanding the Major Factors

### 5.1 Economic Environment Factors

**Country / Market** (unit exposure for all stocks)
- Returns ≈ average stock returns in that market
- Positive expected returns (equity risk premium)
- Being neutral to this factor = being dollar-neutral

**Industry** (binary: 0 or 1 per stock, one per industry)
- More granular than sector ETFs (GICS sub-industries)
- Industry FMPs have zero style factor exposure (unlike sector ETFs)
- Important: industry factor return ≠ sector ETF return

**Beta** (market sensitivity)
- Estimated via weighted regression (exponential decay, ~4-12 months)
- **Betting Against Beta (BAB)**: low-beta stocks outperform high-beta on risk-adjusted basis
  - Frazzini & Pedersen (2014): constrained investors bid up high-beta stocks
  - Beta factor has **negative** expected returns
- **Beta compression**: in crises, all betas converge → less cross-sectional dispersion
- Z-scoring: raw beta → (beta - mean) / std → benchmark beta = 0

**Volatility**
- Long high-vol, short low-vol → **negative returns** (low-vol anomaly)
- Closely related to beta factor (beta = corr * stock_vol / market_vol)
- Explanations: constrained investors overweight volatile stocks; once you add profitability
  and value, the anomaly may disappear
- For PMs: exposure to beta + vol = cost. Quantify it.

### 5.2 Trading Environment Factors

**Short Interest**
- Metrics: short ratio, short-to-float, days-to-cover, utilization rate, borrow rate
- **Heavily shorted stocks underperform** — the strongest anomaly in the data
- Why: informed/unconstrained investors short overpriced stocks; dispersion of beliefs;
  endogenous risk (short squeezes, crowded exits)
- Short interest factor has very large cumulative negative returns

**Active Manager Holdings (AMH)**
- 13(f) filings: quarterly, long-only holdings of institutional investors with >$100M
- Measures "crowding" — how much hedge funds overlap in their positions
- Risk is **endogenous**: external shock → deleveraging → excess supply → more losses
  → more deleveraging (feedback loop)
- The "deleveraging cycle" is the most dangerous dynamic for crowded positions

### 5.3 Technical Factors

**Momentum** — the most robust anomaly across assets and time
- **Term structure**:
  - 0-1 month: **reversal** (short-term mean reversion)
  - 1-12 months: **continuation** (true momentum)
  - >12 months: **reversal** (long-term mean reversion)
- Industry-level momentum exists too
- **Tail risk**: momentum has fat left tails. Crashes come from the *short side* —
  beaten-down stocks rally violently (2009 rebound, 2016 energy squeeze)
- Merton Model analogy: distressed stock = deep OTM call on firm assets → high gamma
- A successful PM naturally accumulates momentum exposure (long winners, short losers)
  → must manage this actively

### 5.4 Valuation Factors

**Value** (book-to-price, cash-flow-to-price, earnings-to-price, EBITDA-to-EV, dividend yield)
- Fama-French: book-to-price (BTOP) is their preferred metric (more stable over time)
- Value stocks are riskier (high fixed costs, unproductive capital, cyclical sensitivity)
- **Gordon's formula**: CF/P = (rate of return - growth) / payout → high CF/P = low growth
- Behavioral explanation: investors overshoot growth estimates (Lakonishok, Shleifer, Vishny)
- Volatility and drawdowns of valuation factors are small → more about strategic portfolio
  positioning than short-term tactical management

**Profitability** (ROE, ROA, margins)
- High profitability → low vol, low beta → **positive** expected returns
- Related to the "quality" factor in modern parlance

### Key Takeaway Table

| Factor | Expected Return | Risk Behavior |
|--------|----------------|---------------|
| Market/Country | Positive | Market risk premium |
| Beta (BAB) | Negative | Compressed in crises |
| Volatility | Negative | Related to beta |
| Short Interest | Strongly negative | Squeeze risk, crowding |
| AMH (crowding) | Variable | Endogenous deleveraging |
| Momentum (1-12m) | Positive | Fat left tail, crash risk |
| Short-term reversal | Positive | Mean reversion |
| Value | Modestly positive | Cyclical, small vol |
| Profitability | Positive | Defensive characteristics |
