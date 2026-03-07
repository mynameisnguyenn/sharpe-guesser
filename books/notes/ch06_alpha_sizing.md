# Chapter 6: Use Effective Heuristics for Alpha Sizing

## The Big Idea
Alpha signals alone are not enough. Renaissance Technologies had alpha early on,
but their Sharpe went from 2 to 6+ only after they nailed portfolio construction.
For fundamental investors, the same principle applies at a slower frequency.

## 6.1 Sharpe Ratio

**Definition**: SR = mean(returns) / vol(returns)

- Practitioners use raw returns (not excess of risk-free rate)
- Information Ratio (IR) = SR computed on *residual* (idiosyncratic) returns
- SR measures return per unit of *risk*, not per unit of *capital*
- If SR is high but vol/GMV is low, you can lever up to increase dollar PnL
- Imperfect (doesn't distinguish upside/downside risk) but universally used

**Converting daily to annual**:
```
SR_annual = sqrt(252) * SR_daily
```

## 6.2 Estimating Expected Returns

Three requirements for expected return inputs:
1. **Idiosyncratic** — strip out industry/market returns. If your $53→$74 forecast
   includes 15% industry return, your idio alpha is 25%, not 40%.
2. **In expectation** — probability-weighted across scenarios, not just the base case.
3. **Common horizon** — scale all forecasts to the same time frame.
   If horizon is 3 months: alpha = r_idio / T.

**Procedure**:
1. Set investment horizon T and price forecast
2. Compute total return = (price forecast / current price) - 1
3. Subtract industry returns over same horizon
4. Divide by T to get annualized alpha

## 6.3 Sizing Rules (from simplest to most complex)

| Rule | Formula (NMV) | Performance |
|------|--------------|-------------|
| **Proportional** | NMV = kappa * alpha | **Best** |
| **Risk Parity** | NMV = kappa * alpha / sigma | Good |
| **Mean-Variance** | NMV = kappa * alpha / sigma^2 | Worst |
| **Shrinked MV** | NMV = kappa * alpha / [p*sigma^2 + (1-p)*sigma_sector^2] | Depends on shrinkage |

Scale kappa so that total GMV or total vol hits your target.

### Why Proportional beats Mean-Variance (the big surprise)

The problem is **alpha estimation error** combined with **volatility dispersion**.

- MV sizes positions proportional to alpha/sigma^2
- Low-vol stocks get much larger positions
- When you're wrong about alpha on a low-vol stock, MV amplifies the mistake
- Example: universe with half 40% vol and half 10% vol stocks, 5% alpha error
  - MV misallocation: (1/2)*0.05/0.10^2 + (1/2)*0.05/0.40^2 = 2.65
  - Uniform vol misallocation: 0.05/0.25^2 = 0.8
  - MV is 3.3x worse!

**Simulation results** (Russell 3000, 1998-2019):
- Proportional: SR = 1.61 (Gaussian signals), 1.29 (buy/sell)
- Risk Parity: SR = 1.46, 1.16
- MV: SR = 1.07, 0.83
- MV loses 34-35% of Sharpe vs. Proportional!

Using *realized* (perfect foresight) volatilities barely helps MV → the problem
is alpha error, not vol estimation error.

## 6.4 From Ideas to Positions (Factor-Neutral Sizing)

Raw proportional sizing will have factor exposures. To neutralize:

1. Regress alpha vector against factor loadings matrix B: `alpha = Bx + epsilon`
2. The residual epsilon is your factor-neutral alpha
3. Standardize: `a_i = epsilon_i / sum(|epsilon|)`
4. Set NMV_i = a_i * target_GMV

This is a simple linear regression — implementable in Excel.

## 6.5 Volatility Targeting (Time-Series)

**Target portfolio vol over time, not GMV.**

Why:
- Volatility is *persistent* and *predictable* (high vol today → high vol tomorrow)
- PnL is NOT persistent (no momentum in fundamental alpha)
- So: reduce GMV when vol spikes to keep dollar vol constant
- This improves SR without sacrificing expected PnL

**Simulation results**: vol targeting beats GMV targeting consistently across:
- All sectors (biggest improvement: financials +11%, TMT +10%)
- All portfolio breadths (50, 100, 200 stocks)
- Both signal types (Gaussian and buy/sell)
- Both horizons (monthly and quarterly forecasts)

**Key insight**: Vol targeting improves SR by 3-11% across sectors.
