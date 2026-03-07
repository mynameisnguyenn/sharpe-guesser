# Analysis Playbook — Paleologo in Practice

Practical analyses to run on firm portfolio data using Omega Point, MSCI Barra,
and Wolfe risk models. Organized by the book's framework.

Data sources assumed:
- **Omega Point**: factor decomposition, performance attribution, exposure monitoring
- **MSCI Barra**: factor loadings, covariance matrices, factor returns
- **Wolfe**: alternative risk model (useful for cross-checking Barra)
- **Internal**: daily positions (NMV by ticker), daily PnL, trade logs

---

## Analysis 1: Where Is Your PnL Coming From?
**Book reference**: Chapters 3, 8

**Question**: How much of each PM's PnL is genuine stock-picking skill vs. riding
factors they don't control?

**Data**: Export from Omega Point — daily PnL attribution (total, factor, idio) by PM.

**What to look for**:
- Plot cumulative total PnL, factor PnL, and idio PnL on the same chart
- If the lines move together, most PnL is factor-driven — not skill
- Drill into factor PnL: is it market beta? Momentum? Industry tilts?
- Compare total Sharpe to idio Sharpe for each PM

**Interpretation**:
- Total SR = 1.5 but idio SR = 0.5 → PM is riding the market, expensive index fund
- Total SR = 0.8 but idio SR = 2.0 → excellent picker buried under factor drag (fixable)
- If style PnL is consistently negative, check which factors are responsible (Analysis 4)

**Action**: PMs with high idio SR but low total SR need better factor management.
PMs with high total SR but low idio SR need honest conversations about their edge.

---

## Analysis 2: Is the Book Clean? (%Idio Variance Over Time)
**Book reference**: Chapter 7

**Question**: What fraction of portfolio risk is coming from stock-specific bets
vs. factors?

**Data**: Omega Point risk decomposition — daily %idio variance by PM and for the fund.

**What to look for**:
- Time series of %idio var. Is it stable? Does it spike during certain periods?
- Is it consistently above 75%? (Paleologo's minimum recommendation)
- Does it drop below 70%? When? What caused it?
- Compare Barra vs. Wolfe — if they disagree by more than 5-10 points, investigate

**Interpretation**:
- Steady 80-90% → well-managed book
- Drops to 60% during vol spikes → factor risk is growing when it's most dangerous
- Slow drift downward over months → PM is accumulating factor exposure passively

**Action**: If %idio var drops below 75%, trigger the tactical factor risk process
(Analysis 4). If Barra and Wolfe disagree significantly, understand why — different
factor definitions can cause this, and neither model is "right."

---

## Analysis 3: Selection, Sizing, and Timing Decomposition
**Book reference**: Chapter 8

**Question**: Is the PM good at picking stocks, sizing positions, or timing when
to be bigger?

**Data**: Daily position-level NMV and daily idiosyncratic returns (from Omega Point
or computed as total return minus factor attribution).

**What to do**:

Step 1 — Build the Cross-Sectional Equalized (XSE) portfolio:
- For each date, take the actual positions
- Replace every NMV with the same absolute value (keeping the sign)
- Scale so total GMV equals the actual portfolio's GMV on that date
- Compute idio PnL of this hypothetical portfolio
- Compare XSE idio Sharpe to actual idio Sharpe
- If XSE wins → PM has negative sizing skill (big bets underperform)

Step 2 — Build the XSTSE portfolio:
- Same as XSE but also equalize across time (constant GMV every day)
- PnL(XSTSE) = pure selection PnL
- PnL(XSE) - PnL(XSTSE) = timing PnL
- PnL(actual) - PnL(XSE) = sizing PnL

Step 3 — Do this separately for longs and shorts:
- A PM might have sizing skill on shorts but not longs (or vice versa)
- The prescription differs: equalize where skill is negative, keep sizing where positive

**Interpretation**:
- Positive selection + negative sizing → equalize positions, keep stock picks
- Positive selection + negative timing → keep GMV constant over time
- Positive everything → rare, protect it

**Action**: This is a quarterly review conversation. Present the decomposition to
each PM and discuss. Not punitive — diagnostic.

---

## Analysis 4: Factor Exposure Heatmap
**Book reference**: Chapters 5, 7

**Question**: Which factor exposures are largest, and are they intentional?

**Data**: Omega Point factor exposures — daily dollar exposure by factor, by PM.

**What to do**:
- Pull daily exposures for the major style factors: momentum (short and medium term),
  beta, volatility, value, profitability, short interest, HF ownership
- Create a heatmap: rows = factors, columns = time, color = dollar exposure
- Separately, compute each factor's contribution to total variance (%var)

**What to look for**:
- Large persistent exposures → are they intentional?
- Exposures that grow over time → passive accumulation (especially momentum)
- Exposures that spike during earnings season → temporary, but monitor
- Any single factor contributing more than 5% of total variance

**Interpretation**:
- $300M momentum exposure contributing 5% of variance → significant, needs managing
- $50M value exposure contributing 0.5% of variance → negligible, ignore
- HF ownership exposure growing → the book is getting more crowded (deleveraging risk)

**Action**: For any factor exceeding single-factor limits, identify the stocks with
highest MCFR (marginal contribution to factor risk) and discuss trades with the PM.

---

## Analysis 5: Momentum Exposure Drift
**Book reference**: Chapters 5, 7

**Question**: Is the portfolio passively accumulating momentum exposure from
successful stock picks?

**Data**: Omega Point — daily momentum factor exposure (both short-term and
medium-term) by PM.

**What to do**:
- Plot momentum exposure over time alongside cumulative idio PnL
- Check the correlation: does momentum exposure grow when idio PnL is positive?
- Compute the ratio: momentum $var / total $var over rolling windows

**Why this matters**:
- A PM who is long stocks that keep going up naturally gets long momentum
- A PM who is short stocks that keep going down naturally gets short momentum
- Both increase momentum exposure without any active decision
- Momentum is the factor most likely to crash violently (2009, 2016, 2020)

**Interpretation**:
- Momentum exposure and idio PnL are correlated → this is the natural accumulation
  Paleologo warns about. The PM's success is creating a hidden risk.
- Momentum exposure is stable while idio PnL grows → the PM or risk team is
  actively managing it. Good.

**Action**: Set a maximum momentum exposure limit. When the book approaches it,
the PM needs to either trim winners, cover shorts that have fallen, or add
counter-momentum positions as hedges.

---

## Analysis 6: Would Equal-Weighting Improve Sharpe?
**Book reference**: Chapter 6

**Question**: Is the PM's sizing adding or destroying value?

**Data**: Daily positions (NMV) and daily idio returns by stock.

**What to do**:
- Compute actual portfolio idio Sharpe
- Construct equal-weighted version (same stocks and sides, same GMV, equal sizes)
- Compute equal-weighted portfolio idio Sharpe
- Do this separately for longs and shorts
- Also compare proportional-to-alpha sizing if alpha estimates are available

**What to look for**:
- Equal-weighted SR > actual SR → sizing is destroying value
- The gap on longs vs. shorts separately — skill may differ by side
- How concentrated the actual portfolio is: effective N vs. actual N

**Interpretation**:
- If equal-weighting wins by more than 10%: strong evidence that the PM should
  flatten position sizes. This is the single easiest performance improvement available.
- If actual sizing wins on shorts but loses on longs: equalize longs, keep short sizing

**Action**: Present to the PM as a "what-if." Not "you should equal-weight" but
"here's what the data says about your sizing decisions over the past year."

---

## Analysis 7: How Concentrated Is the Book? (Effective N)
**Book reference**: Chapter 8

**Question**: How diversified is the portfolio really, accounting for position
size differences?

**Data**: Daily positions (NMV by ticker).

**What to do**:
- For each date, compute: effective_N = 1 / sum(pct_GMV_i^2)
- Where pct_GMV_i = |NMV_i| / GMV for each stock
- Track over time, separately for longs and shorts
- Compare to actual number of positions

**What to look for**:
- A 70-stock portfolio with effective N of 30 → very concentrated
- A 70-stock portfolio with effective N of 60 → well diversified
- Effective N declining over time → one or two positions are growing to dominate

**Interpretation**:
- Using the fundamental law: IR = (2 * hit_rate - 1) * sqrt(252 * effective_N)
- If effective N drops from 60 to 30, the PM needs 0.7% higher hit rate to maintain
  the same IR. That's a lot.
- Low effective N also means single-position risk is elevated

**Action**: Set a minimum effective N threshold. If violated, the PM needs to
either trim the largest positions or add new names.

---

## Analysis 8: Vol Targeting Backtest
**Book reference**: Chapter 6

**Question**: Would the portfolio have performed better if GMV had been scaled
to maintain constant dollar volatility?

**Data**: Daily portfolio PnL, daily predicted portfolio vol (from Omega Point or
Barra), daily GMV.

**What to do**:
- Compute the ratio: predicted_dollar_vol / target_vol on each day
- The vol-targeted GMV would be: actual_GMV * (target_vol / predicted_dollar_vol)
- Scale daily PnL by (vol-targeted GMV / actual GMV) to get hypothetical vol-targeted PnL
- Compare Sharpe ratios: actual vs. vol-targeted

**What to look for**:
- Does vol-targeted Sharpe exceed actual Sharpe? By how much?
- Look at the periods where they diverge: typically during vol spikes (March 2020,
  late 2018, early 2016)
- Vol targeting should reduce drawdowns during stress periods

**Interpretation**:
- Vol-targeted SR > actual SR by 5-10% → consistent with Paleologo's findings,
  worth implementing
- Little difference → the PM is already implicitly vol targeting (cutting risk in
  high-vol periods)
- Vol-targeted SR is lower → unusual, investigate. Might mean PnL is positively
  correlated with vol (the PM does better in volatile markets)

**Action**: If the backtest confirms improvement, propose a formal vol-targeting
overlay. Start with idio dollar vol targeting (not total vol — keep the separation
of concerns).

---

## Analysis 9: Stop-Loss Efficiency
**Book reference**: Chapter 9

**Question**: Given each PM's realized Sharpe, how much performance is the
stop-loss policy costing?

**Data**: Each PM's annualized idio Sharpe, annualized dollar vol, stop-loss
threshold (from fund policy).

**What to do**:
- Compute the ratio: stop_loss_threshold / annualized_dollar_vol for each PM
- Look up efficiency in Paleologo's table (or interpolate):

| Loss/Vol | SR=0.5 | SR=1.0 | SR=1.5 | SR=2.0 |
|----------|--------|--------|--------|--------|
| 1.0x     | 33%    | 47%    | 64%    | 78%    |
| 1.4x     | 58%    | 75%    | 88%    | 96%    |
| 1.8x     | 76%    | 90%    | 97%    | 99%    |
| 2.0x     | 83%    | 93%    | 98%    | 100%   |

- Compute firm-level impact: effective firm SR after stops

**Interpretation**:
- PM with SR=1.5 and loss/vol=1.5x → 88% efficiency. Reasonable cost.
- PM with SR=0.7 and loss/vol=1.2x → ~55% efficiency. Very expensive insurance.
- If two PMs have the same stop but very different Sharpes, the weak PM is paying
  much more for the same insurance

**Action**: Present to CIO/CRO. Consider whether stops should be differentiated
by PM track record, or whether weak PMs need wider stops (counterintuitive but
the math supports it for certain ranges).

---

## Analysis 10: Crowding / Overlap with Hedge Fund Consensus
**Book reference**: Chapter 5 (AMH section)

**Question**: How much does the portfolio overlap with the crowded hedge fund trade?

**Data**: Omega Point crowding analytics (if available), or 13F-derived HF ownership
data, or Wolfe's crowding factor.

**What to do**:
- Compute portfolio exposure to the HF ownership / crowding factor
- List the top 10 long positions and check their ownership by hedge funds
  (available via Bloomberg, Omega Point, or 13F aggregators)
- Compare the portfolio's factor profile to the "average hedge fund" profile
  (some vendors provide this as a reference portfolio)

**What to look for**:
- High positive exposure to the crowding factor → you own what everyone owns
- Multiple top-10 longs appearing in "most popular hedge fund stocks" lists
- Crowding exposure increasing over time

**Why this matters**:
- Crowded positions are vulnerable to the deleveraging cycle (Chapter 5)
- When one fund blows up and liquidates, all the crowded names sell off together
- Your idiosyncratic thesis on a stock doesn't protect you from the crowding factor

**Interpretation**:
- High crowding + high single-name concentration → most dangerous combination
- High crowding + well diversified → somewhat protected by diversification
- Low crowding → your ideas are differentiated from consensus. Good signal.

**Action**: Not necessarily "avoid crowded names" — your thesis may be right.
But quantify the crowding risk. Know that if the crowding factor sells off 5%,
here's what it costs you. Make it a conscious bet, not an accidental one.

---

## Cross-Model Comparison (Bonus Analysis)
**Book reference**: Chapter 4 FAQ

**Question**: How much do Barra and Wolfe disagree on your portfolio's risk?

**Data**: Risk decomposition from both models for the same portfolio on the same dates.

**What to do**:
- Compare %idio var from Barra vs. Wolfe
- Compare top factor exposures: do they agree on which factors dominate risk?
- Compare predicted portfolio vol: how far apart are they?

**What to look for**:
- %idio var disagreement of 5-10 points is normal (different factor definitions,
  different estimation methods)
- If they disagree on the dominant factor (Barra says momentum, Wolfe says beta),
  investigate which model's factor definition better matches your portfolio
- If predicted vol differs by more than 20%, one model may be wrong for your
  universe

**Interpretation**:
- Agreement → high confidence in the risk estimate
- Disagreement → you're in a gray zone. Don't over-optimize to either model.
  Use the more conservative estimate for risk limits.

**Action**: Run both models monthly. When they diverge significantly, flag it.
The divergence itself is information — it means the portfolio is in a region
where models are uncertain.

---

## Suggested Sequence

If you're starting from scratch, run these in order:

1. **Analysis 1** (factor vs. idio PnL) — the foundational decomposition
2. **Analysis 2** (%idio var) — is the book clean?
3. **Analysis 4** (factor exposure heatmap) — where is factor risk coming from?
4. **Analysis 7** (effective N) — how diversified are you really?
5. **Analysis 6** (equal-weighting comparison) — is sizing helping or hurting?
6. **Analysis 3** (selection/sizing/timing) — the deep performance decomposition
7. **Analysis 5** (momentum drift) — the most common passive risk accumulation
8. **Analysis 8** (vol targeting) — would this improve risk-adjusted returns?
9. **Analysis 10** (crowding) — how exposed are you to the consensus trade?
10. **Analysis 9** (stop-loss efficiency) — is the insurance priced right?
11. **Cross-model comparison** — how much do you trust your risk numbers?
