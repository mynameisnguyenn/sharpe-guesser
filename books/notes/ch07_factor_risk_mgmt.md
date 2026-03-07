# Chapter 7: Manage Factor Risk

## 7.1 Tactical Factor Risk Management

### Why manage factor risk?
1. You have **no edge** in factor investing — specialized quant players barely break even
2. **Separation of concerns** — isolate idio PnL for better diagnosis
3. **Fiduciary duty** — investors aren't paying you to replicate the market

### Risk Decomposition Table

The main diagnostic tool. For each factor, compute:

| Column | What it means | How to use it |
|--------|--------------|---------------|
| **%Var** | % of total variance from this factor | Focus attention on largest contributors |
| **$Exposure** | Dollar exposure to factor (sum of loading * NMV) | Scenario analysis: "if factor moves X, I lose Y" |
| **$Vol** | Annualized dollar volatility from this factor | Factor risk ≈ exposure * factor vol |
| **MCFR** | Marginal Contribution to Factor Risk | Which stock trades reduce factor risk most? |

**Example**: A tech portfolio has 85% idio var, 15% factor var.
- Style: 7.6% of total var (dominated by 12-month momentum at 5.05%)
- Industry: 7.4% of total var
- Action: reduce momentum exposure by identifying stocks with high MCFR

### Per-Stock Action Process
For each stock, check:
1. MCFR — how much does trading this stock reduce factor risk?
2. Factor loading — what's the stock's exposure to the target factor?
3. Conviction — does the trade align with your fundamental thesis?

Actions: BUY (increases positions that reduce risk), REDUCE, SELL SHORT, COVER, DO NOT TRADE

### Tactical Portfolio Construction Procedure
1. Start with fundamental theses → convert to dollar positions (Ch 6)
2. Perform risk decomposition
3. If %idio var too low → identify responsible factors
4. Identify largest factor contributors + exposures exceeding limits
5. Identify stock trades that reduce factor risk AND align with conviction
6. Execute trades

### When to Use Optimization
- Factor "rotations" (momentum crash, short squeeze)
- Alpha is strong → momentum/beta exposure grows naturally
- Earnings season → temporary factor risk spikes
- Custom factors (ESG scores, etc.) need managing

Optimization formulation:
```
min (transaction cost)
s.t. final factor exposures bounded
     custom exposures bounded
     factor risk bounded
     GMV bounded below
```

## 7.2 Strategic Factor Risk Management

### Setting %Idio Variance Limits

**Formula**: `SR = IR * sqrt(%idio_var)`

| %Idio Var | IR Degradation |
|-----------|---------------|
| 95% | 2.5% |
| 90% | 5.1% |
| 85% | 7.8% |
| 80% | 10.6% |
| 75% | 13.4% |

**Rule of thumb**: every 5-point drop in %idio var costs ~2.5% performance.

**Recommendation**: don't go below 75% idio var. Below 70% is almost never acceptable.

But 100% idio var has costs too:
- Cognitive cost of continuous neutralization
- Transaction costs from factor-adjusting trades
- May distort sizing decisions
- Risk model may be wrong (±5-10 points across vendors)

### Market Exposure

**Optimal market allocation**:
```
market_var / total_var = 1 / (1 + (SR_port / SR_market)^2)
```

- If SR_port = 2 and SR_market = 0.5: optimal market var = 1/(1+16) = 6% → basically neutral
- If SR_port = 0.5 and SR_market = 0.5: optimal market var = 50% → significant beta

**Practical**: broad portfolios (>1000 stocks) with SR>1.5 should be market-neutral.
Narrow portfolios (<50 stocks) with mediocre SR benefit from some market exposure.

### Single-Position Limits
- Max position GMV should increase with conviction skill, decrease with portfolio breadth
- Prevents concentration risk from blowing up the portfolio

### Single-Factor Exposure Limits
- Based on loss tolerance and worst-case factor returns
- Example: if momentum can lose 10% in a bad month, and you can tolerate $5M loss
  from any single factor, cap momentum exposure at $50M

## Hedging Programs (Advanced)
For PMs who need higher productivity:
- Automated hedging generates perfect factor hedges
- Adds "hedge basket" stocks to the portfolio
- Trade-off: more stocks to manage, but factor risk stays controlled
