# Chapter 11: Appendix — Key Formulas

## 11.1 Factor Model Master Equation
```
r_t = alpha + B_t * f_t + epsilon_t
```
- r_t: n-vector of asset returns
- alpha: n-vector of expected returns
- B_t: n x m loadings matrix
- f_t: m-vector of factor returns ~ N(0, Omega_f)
- epsilon_t: n-vector of idio returns ~ N(0, Omega_epsilon), diagonal

**Asset covariance matrix**: `Omega_r = B * Omega_f * B' + Omega_epsilon`

## 11.1.2 Factor-Mimicking Portfolios
Cross-sectional weighted least squares:
```
min (r - Bf)' W (r - Bf)
```
Ideal W = inverse of idio covariance. In practice: W_ii = sqrt(market_cap_i).

Solution: `f_hat = (B'WB)^{-1} B'Wr`

The FMP weights for factor j: `w_j = (B'WB)^{-1} B' W` (j-th row)

## 11.2 Fundamental Law of Active Management

For a portfolio with hit rate p and N_e effective stocks:
```
IR = (2p - 1) * sqrt(252 * N_e)
```

Effective number of stocks: `N_e = 1 / sum(w_i^2)` where w_i = |NMV_i| / GMV

## 11.3 Why Proportional Sizing Beats MV

When alpha has estimation error epsilon with variance tau^2:
- Proportional: Sharpe degradation = O(tau^2)
- MV: Sharpe degradation = O(tau^2 * dispersion_of_1/sigma^2)

If sigma varies across stocks, the 1/sigma^2 weighting amplifies errors on low-vol stocks.

Shrinkage toward sector variance reduces this but doesn't eliminate it.

## 11.4 Factor-Neutral Sizing

Given alpha vector a and loadings B:
1. Regress: a = Bx + epsilon (OLS)
2. Residual epsilon is factor-orthogonal
3. Normalize: NMV_i = epsilon_i / sum(|epsilon_j|) * target_GMV

This minimizes ||a* - a||^2 subject to B'a* = 0 and sum(|a*|) = GMV.

## 11.7 Tactical Optimization Formulation
```
min sum_i |trade_i| * tcost_i
s.t. |B'(h + trade)| <= exposure_limits   (factor exposure bounds)
     (h + trade)' Omega_f (h + trade) <= max_factor_var  (factor risk bound)
     sum(|h_i + trade_i|) >= min_GMV       (minimum GMV)
```
Where h = current holdings, trade = proposed trades.

## 11.9 Optimal Event Trading

For earnings trade with:
- alpha = expected return on event day
- V = daily dollar volume
- sigma = daily volatility
- T = days until event
- C = market-specific constant

```
optimal_trade_size = C * alpha * V * T / (2 * sigma)
```

Trade at constant VWAP participation rate from now until event.
Liquidate at same rate after event.

## Key Mathematical Identities

**Variance of portfolio**: `Var(portfolio) = w' Omega w`

**Marginal contribution to risk**: `MCFR_i = d(factor_vol) / d(NMV_i)`

**Volatility scaling**: `vol_annual = vol_daily * sqrt(252)`

**Diversification**: `vol(equal-weight N stocks) = sigma / sqrt(N)`
