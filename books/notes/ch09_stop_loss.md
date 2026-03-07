# Chapter 9: Manage Your Losses (Stop-Loss)

## How Stop-Loss Works

A PM agrees to a maximum tolerable loss (measured from high watermark).
When the threshold is reached: partial or total liquidation.

### Two Variants
1. **Single-threshold**: liquidate everything at X% loss
2. **Two-threshold**: cut GMV by 50% at X/2% loss, liquidate fully at X% loss
   - Encourages discipline; results are nearly identical to single-threshold

### Two Types of Stops
- **Stop-and-Shutdown**: strategy is permanently closed
- **Stop-and-Restart**: portfolio liquidated, but pro-forma performance monitored;
  re-capitalized if performance recovers past a hurdle

## Arguments FOR Stop-Loss

1. **Everyone has one** — even firms that claim they don't. The question is whether
   it's transparent or arbitrary. Transparent is better.

2. **Offsets the PM's call option** — PM compensation = call option on portfolio.
   Higher vol = higher option value → PM has incentive to take excess risk.
   Stop-loss caps the downside, reducing this misaligned incentive.

3. **Portfolio insurance** — equivalent to buying an OTM put on the portfolio.
   - Grossman-Zhou optimal policy: reduce GMV proportionally as drawdown increases
   - Two-threshold rule approximates this behavior
   - More volatile strategies must reduce earlier but retain more optionality

## Arguments AGAINST Stop-Loss

1. **Transaction costs** — de-grossing and re-grossing adds ~5% extra turnover.
   Individual trades during liquidation are much larger than routine trades.

2. **Opportunity cost** — this is the bigger problem. By cutting capital in drawdown,
   you miss the recovery. If alpha is mean-reverting, stop-loss actively hurts.

## The Efficiency Trade-Off

**Efficiency** = realized SR with stop-loss / true SR of PM

Simulations (5-year horizon, heavy-tailed returns):

| Loss/Vol Ratio | SR=0.5 | SR=1.0 | SR=1.5 | SR=2.0 |
|---------------|--------|--------|--------|--------|
| 1.0 | 33% | 47% | 64% | 78% |
| 1.2 | 47% | 62% | 79% | 88% |
| 1.4 | 58% | 75% | 88% | 96% |
| 1.6 | 69% | 85% | 93% | 98% |
| 1.8 | 76% | 90% | 97% | 99% |
| 2.0 | 83% | 93% | 98% | 100% |

**Reading**: A PM with true SR=1 and stop-loss at 1.5x vol retains only 75% efficiency.
Four such PMs with uncorrelated returns: firm SR drops from 2.0 to 1.5.

**Key insights**:
- High-SR PMs barely affected (loss/vol=1.4 costs a SR=2 PM only 4%)
- Low-SR PMs pay dearly (same threshold costs a SR=0.5 PM 42%)
- The steeper the stop, the more you pay
- Single vs. two-threshold difference is negligible

## Practical Guidelines
- Stop-loss at 1.5-2.0x annualized vol is a reasonable starting point
- Higher SR → can afford tighter stop (less likely to trigger)
- Lower SR → wider stop needed (but still eventually triggers)
- For a PM with SR=1, set stop at 1.5-2.0x vol → efficiency = 75-93%
