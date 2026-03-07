# Chapter 10: Set Your Leverage Ratio

## Definition
```
Leverage = GMV / AUM
```

This chapter is most relevant if you're starting a fund or managing fund-level risk.

## Key Assumptions (simplified model)
1. Negligible factor variance (market-neutral portfolio)
2. All stocks have same idio vol (sigma)
3. Portfolio has Sharpe Ratio s
4. n stocks with equal position sizes
5. Zero borrow cost

## The Relationships

**Portfolio dollar vol / GMV**:
```
sigma_port / GMV = sigma / sqrt(n)
```
Example: 70 stocks, 20% single-stock idio vol → portfolio vol/GMV = 2.4%
At GMV = $1B → dollar vol = $24M annually

**Expected PnL**:
```
E[PnL] = SR * sigma_port = SR * sigma * GMV / sqrt(n)
```

**Return on AUM**:
```
E[return on AUM] = SR * sigma * L / sqrt(n)
```
Where L = leverage ratio

## Upper Bound on Leverage

Leverage is bounded by:
1. **Return target** — need enough leverage to generate acceptable returns
2. **Maximum acceptable loss** — stop-loss constraint limits how much you can lever

The fundamental formula tying it together:
```
L_max = max_loss / (k * sigma / sqrt(n))
```
Where k is a multiple related to your stop-loss threshold.

## Practical Implications

- **Breadth helps leverage**: more stocks → lower vol/GMV → can lever more
- **Lower single-stock vol helps**: same reason
- **Higher SR allows tighter stops**: which allows more leverage
- **Interaction**: breadth, vol, SR, and stop-loss threshold are all connected

## Decision Framework
1. Set your target return on AUM
2. Set your maximum acceptable loss (stop-loss threshold)
3. Given your portfolio's breadth and stock-level vol, compute the leverage range
4. Check that your leverage is compatible with both the return target AND the loss limit
5. If not → either increase breadth, reduce single-stock vol, or accept lower returns
