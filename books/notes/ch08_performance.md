# Chapter 8: Understand Your Performance

## 8.1 Factor Attribution

"The Earth rotates at 67,000 mph. My running speed is in the tens of thousands.
Should I brag?" — Factor PnL is the sun's rotation. Idio PnL is your actual speed.

### Performance Attribution
```
Total PnL = Factor PnL + Idio PnL
Factor PnL = sum over factors of (dollar_exposure_j * factor_return_j)
```

**Example**: A net-long PM has total SR=1.56. Looks great! But:
- Idio SR = 2.4 (excellent stock picking)
- Style PnL is deeply negative (especially momentum exposure)
- Net of market: performance is mediocre
- Diagnosis: great alpha, bad factor management → fixable!

## 8.2 Idiosyncratic Decomposition: Selection, Sizing, Timing

### The Decomposition
```
PnL(Idio) = Selection PnL + Sizing PnL + Timing PnL
```

**Selection**: Are you directionally right? (long winners, short losers)
**Sizing**: Are you sizing the best ideas bigger?
**Timing**: Are you increasing risk when your ideas are working?

### How to Measure

1. **Cross-Sectional Equalized (XSE)**: Replace all NMVs with equal sizes (keeping
   the sign). Same GMV per date, but uniform sizing. Compare PnL.
   - If XSE SR > raw SR → you have negative sizing skill (your big bets underperform)
   - If XSE SR < raw SR → you have positive sizing skill

2. **Cross-Sectional Time-Size Equalized (XSTSE)**: Equal sizes across stocks AND time.
   Same GMV every day. Compare PnL.
   - PnL(XSE) - PnL(XSTSE) = timing PnL
   - PnL(XSTSE) = pure selection PnL

### Practical Notes
- Set a minimum GMV threshold (e.g., $1M) — exclude tiny residual positions
- Analyze idio performance only (use risk model to strip factors)
- Ignore transaction costs initially (refine later)
- For vol-based analysis, equalize dollar idio vol instead of GMV

### Empirical Finding
Most PMs have:
- Positive **selection** skill (they pick the right stocks)
- Little to no **timing** skill (they can't time when to be bigger)
- Mixed **sizing** skill (often slightly negative)

**Action**: If you have negative sizing skill → equalize positions.
If you have negative timing skill → keep GMV constant over time.

## 8.3 Diversification and Hit Rate

**The Fundamental Law of Active Management** (simplified):
```
IR = [2 * hit_rate - 1] * sqrt(252 * effective_number_of_stocks)
```

| Hit Rate | 70 Stocks | 200 Stocks | 3000 Stocks |
|----------|-----------|------------|-------------|
| 50.5% | IR = 1.3 | IR = 2.2 | IR = 8.7 |
| 51.0% | IR = 2.7 | IR = 4.5 | IR = 17.3 |
| 52.0% | IR = 5.3 | IR = 9.0 | IR = 34.7 |

**Effective number of stocks**: 1 / sum(pct_GMV_i^2)
- Equal-weighted 100 stocks → effective N = 100
- If one stock is 50% of GMV → effective N much less

**Key insight**: You don't need high accuracy. 50.5% hit rate with 70 stocks = IR of 1.3.
This explains why stat arb works: thousands of stocks * 50.5% accuracy = very high IR.

**But**: diversification has diminishing returns for fundamental PMs.
Beyond ~100 stocks, accuracy drops as coverage expands. Optimal breadth is where
marginal cost of analysis = marginal benefit of diversification.

## 8.4 Trading Around Events (Earnings)

**Optimal trade size for earnings**:
```
optimal_size = C * alpha * Volume * T / (2 * sigma)
```
Where T = days until event, sigma = daily vol, Volume = daily dollar volume.

Rules of thumb:
- Size at event proportional to expected return
- Smaller if transaction costs higher
- Smaller if less time to build
- Liquidate after event (consumes risk budget)
- Trade at constant % of volume (VWAP)
- For concentrated portfolios, risk penalty reduces optimal size further

25-50% of a PM's PnL can come from earnings-related positions.

## 8.5 Alternative Data

Framework for evaluating new data:
1. **Feature generation**: compress raw data into per-stock characteristics
2. **Feature transformation**: differences, growth rates, z-scores, interactions
3. **Orthogonalization**: regress features against risk model loadings — keep the residual
4. **Cross-sectional regression**: check if orthogonalized features predict residual returns
5. **Performance metrics**: expected return, SR, stability of loadings

Example: Short Interest was "alternative data" before being included in risk models.
It has strong predictive power but is still not in most commercial models (data recency
issues, debated risk explanatory power).

## Takeaway Checklist
1. Decompose PnL into factor vs. idio
2. If factor PnL is large → your factor risk is too high
3. Decompose idio into selection + sizing + timing
4. If positive sizing/timing skill → lean into it
5. If negative → equalize in that dimension
6. Diversify as much as possible without degrading hit rate
7. Trade events at VWAP, size proportional to alpha * volume * time / vol
