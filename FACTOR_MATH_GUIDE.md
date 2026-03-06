# The Math Behind Factor Models — A Practical Guide

You read risk reports every day. You see beta, R-squared, factor exposures, tracking
error. You know what the numbers roughly mean, but you're not sure *why* the math
works the way it does. This guide fixes that.

No code here. For implementation, see `modules/module_3_factor_models.py` and
`factor_dashboard.py`.

---

## 1. What Is a Factor Model?

Your risk report already breaks down PnL into components: market exposure, sector
tilt, and stock-specific return. That breakdown **is** a factor model. You've been
using one every day.

Think of factors as ingredients in a recipe. A stock's daily return is a mix of:

- How the broad market moved (market factor)
- Whether small-caps outperformed large-caps that day (size factor)
- Whether value stocks beat growth stocks (value factor)
- Something unique to that stock — an earnings surprise, a FDA approval, a CEO tweet

The factor model separates these ingredients so you can see which ones are driving
your portfolio's returns.

> **Key insight:** If factors explain 80% of a stock's movement, only 20% is
> genuinely stock-specific. That 20% is what your PMs are actually being paid to find.

This matters because investors don't pay 2-and-20 for factor exposure they could
get from a Vanguard ETF. They pay for the part that factors *can't* explain.

---

## 2. Beta — Your Stock's Sensitivity to the Market

Beta is the simplest factor concept, and you already use it intuitively.

**Plain English:** If SPY goes up 1% and your stock goes up 1.3%, your stock's
beta is approximately 1.3. It amplifies market moves by 30%.

The formal equation:

```
R_stock = alpha + beta x R_market + epsilon
```

| Piece | What it means |
|-------|---------------|
| **R_stock** | Your stock's return on a given day |
| **alpha** | The intercept — return that isn't explained by the market |
| **beta** | How much the stock moves per 1% market move |
| **R_market** | The market's return (usually SPY) |
| **epsilon** | The residual — random noise, earnings surprises, stock-specific events |

### A Numerical Example

Suppose you run a regression on JPM over the past year and get:

```
R_JPM = 0.02% + 1.15 x R_SPY + epsilon
```

This tells you:

- **Beta = 1.15**: When SPY moves 1%, JPM tends to move 1.15% in the same direction.
  JPM is slightly more volatile than the market.
- **Alpha = 0.02% daily (about 5% annualized)**: JPM earned roughly 5% per year
  *beyond* what its market exposure would predict. (Whether this is statistically
  significant is a separate question — see Section 3.)
- **Epsilon**: The daily noise. Some days JPM moves +2% even though SPY is flat.
  That's epsilon — bank earnings, rate expectations, analyst upgrades.

### Quick Beta Reference

| Beta | Interpretation | Typical stocks |
|------|---------------|----------------|
| > 1.5 | Very aggressive, amplifies market moves | High-growth tech, biotech |
| 1.0 | Moves with the market | Broad index, diversified large-caps |
| 0.5–0.8 | Defensive, dampens market moves | Utilities, consumer staples |
| ~0 | Uncorrelated with market | Market-neutral strategies |
| < 0 | Moves opposite to market | Rare — some tail-hedge instruments |

### Why PMs Have Beta Targets

Your desk monitors beta drift because it's unintended risk. If a PM has a
mandate to run at 0.5 beta and they drift to 1.2, the portfolio is now taking
on market risk the investor didn't sign up for. That's a risk conversation, not
an alpha conversation.

### What Beta Is NOT

Beta measures **systematic risk** — sensitivity to the broad market. It does not
measure total risk. A biotech stock waiting on an FDA ruling might have a beta
of 0.3 (low market sensitivity) but be extraordinarily risky because of the
binary event. Beta only captures the market-driven part of risk.

---

## 3. Alpha — The Holy Grail

Alpha is the return left over after you remove all factor exposure. In the CAPM
equation above, alpha is the intercept — the return your PM generated that the
market didn't hand them for free.

**Here's the uncomfortable truth:** most "alpha" isn't really alpha. It's hidden
factor exposure.

Consider a PM who returned 18% last year and claims to be a great stock picker.
You run a factor model and discover:

- Their portfolio has a beta of 1.3 (the market was up 12%, so beta alone
  contributed ~15.6%)
- They're heavily tilted toward small-cap value stocks (SMB and HML loadings
  are high)
- After accounting for all factor exposures, true alpha is **-1%**

They didn't pick great stocks. They loaded up on factors that happened to
perform well. An ETF portfolio with the same factor tilts would have done
better — and charged 0.03% instead of 2%.

### Is the Alpha Real? The t-Statistic

Finding alpha in a regression doesn't mean it's real. Markets are noisy. You
need to ask: **how confident are we that this alpha isn't just luck?**

That's what the t-statistic tells you.

> **t-statistic in plain English:** "How many standard errors is the alpha
> away from zero?" If it's far from zero, it's unlikely to be luck.

The rule of thumb:

| t-stat | Interpretation |
|--------|---------------|
| < 1.0 | Could easily be noise. No evidence of real alpha. |
| 1.0–2.0 | Suggestive but not convincing. Wouldn't bet the farm on it. |
| > 2.0 | Statistically significant at ~95% confidence. Worth paying attention to. |
| > 3.0 | Very strong evidence. This is rare in live portfolios. |

**Example:** An alpha of 3% with a t-stat of 0.8 is *less* impressive than an
alpha of 1% with a t-stat of 2.5. The first is likely noise with a big number
attached. The second is small but probably real.

This connects directly to Paleologo's emphasis on separating skill from exposure.
Most funds that appear to generate alpha are actually harvesting factor premia
that could be replicated cheaply. True alpha — with statistical significance over
a meaningful time horizon — is rare.

---

## 4. Fama-French: Beyond Just the Market

CAPM uses one factor: the market. Eugene Fama and Kenneth French showed that two
additional factors explain a large chunk of what CAPM misses.

### The Three Factors

**MKT (Market Excess Return)** — same as CAPM beta. How much does your portfolio
move with the broad market?

**SMB (Small Minus Big)** — the size factor. This measures the return difference
between small-cap and large-cap stocks. If your fund is long a basket of $2B
companies and short mega-caps, your SMB loading will be positive. You're making a
bet that small stocks will outperform large stocks.

**HML (High Minus Low)** — the value factor. This measures the return difference
between cheap stocks (high book-to-market) and expensive stocks (low
book-to-market). If your PM loves stocks with low P/E ratios and avoids
high-multiple growth names, your HML loading will be positive.

The regression becomes:

```
R_stock - Rf = alpha + b1 x MKT + b2 x SMB + b3 x HML + epsilon
```

Each coefficient (b1, b2, b3) tells you how exposed the portfolio is to that
factor. Together, they paint a picture of *what kind* of risk the PM is taking.

### Why This Matters: A Worked Example

Imagine Fund A returns 12% in a year. Impressive? Let's decompose it.

| Factor | Fund's Loading | Factor Return | Contribution |
|--------|---------------|---------------|-------------|
| MKT | 1.0 | 10% | 10.0% |
| SMB | 0.3 | 5% | 1.5% |
| HML | 0.4 | 3% | 1.2% |
| **Total factor contribution** | | | **12.7%** |
| **Alpha** | | | **-0.7%** |

The fund returned 12%, but factors alone would have returned 12.7%. The PM
actually **destroyed** 0.7% of value through stock selection. The impressive
headline number was entirely factor exposure — exposure you could replicate
with three ETFs for almost no fee.

This is why Paleologo and every serious allocator decompose returns this way.
The raw return number is meaningless without knowing what factors are underneath.

### Beyond Three Factors

The industry has expanded well beyond Fama-French. Common additional factors:

- **Momentum (MOM)**: stocks that went up keep going up (short-term)
- **Quality (QMJ)**: profitable, stable companies outperform junk
- **Low Volatility**: boring stocks beat exciting ones on a risk-adjusted basis
- **Betting Against Beta (BAB)**: low-beta stocks outperform high-beta stocks

Your risk system likely uses 5-10+ factors. The principle is the same: decompose
returns to see what's factor-driven and what's genuine stock picking.

---

## 5. R-Squared — How Much Is Factor-Driven?

R-squared (R^2) is the percentage of a portfolio's return variation that the
factor model explains. It answers: **how much of what this PM does could you
replicate with factor ETFs?**

```
R^2 = (variance explained by factors) / (total variance of returns)
```

### Interpretation

| R^2 | What it means | Implication |
|-----|--------------|-------------|
| > 90% | Almost entirely factor-driven | Closet indexer. Why pay active fees? |
| 70–90% | Mostly factor-driven with some stock picking | Typical for most long-only managers |
| 40–70% | Meaningful stock-specific component | Active manager with genuine idiosyncratic bets |
| < 40% | Mostly stock-specific | True stock picker or unusual strategy |

### The R-Squared Tension

LPs (limited partners — the investors in your fund) face a dilemma:

- **They want low R^2** because that means they're paying for genuine alpha, not
  factor exposure they could get cheaply.
- **But low R^2 means higher tracking error** — the portfolio will behave very
  differently from any benchmark. That feels uncomfortable, especially during
  drawdowns.

This tension drives a lot of investor conversations. A PM with R^2 of 95% to the
S&P 500 is basically an expensive index fund. A PM with R^2 of 25% is taking
genuinely different bets, which is what LPs want — until it underperforms for
two quarters and they panic.

> **Connect to your daily work:** When your risk report shows high correlation
> to SPY, that's essentially saying R^2 is high. The portfolio is moving with
> the market, not independently of it.

---

## 6. Information Ratio — The Report Card for Active Managers

The Information Ratio (IR) is how the institutional world evaluates active managers.
It answers: **how much alpha are you generating per unit of active risk?**

```
IR = alpha / tracking error
```

Where:
- **Alpha** = annualized excess return over what factors predict
- **Tracking error** = annualized standard deviation of the residuals (the
  day-to-day variability of alpha)

**Plain English:** A PM who generates 3% alpha with 6% tracking error has an IR
of 0.5. A PM who generates 3% alpha with 15% tracking error has an IR of 0.2.
Same alpha, but the first PM delivers it much more consistently.

### Benchmarks

| IR | Rating | How rare |
|----|--------|----------|
| < 0.0 | Negative — destroying value | ~40% of active managers |
| 0.0–0.3 | Below average | Common |
| 0.3–0.5 | Decent | Top quartile territory |
| 0.5–1.0 | Very good | Top decile |
| > 1.0 | Exceptional | Extremely rare over long periods |

An IR above 0.5 sustained over 5+ years is genuinely impressive. Most managers
cluster around 0.0–0.3.

### The Fundamental Law of Active Management

Grinold and Kahn's "Fundamental Law" decomposes the IR into two pieces:

```
IR = IC x sqrt(breadth)
```

| Piece | What it means |
|-------|---------------|
| **IC (Information Coefficient)** | How good your predictions are — the correlation between your forecast and the actual outcome. An IC of 0.05 means your predictions are slightly better than a coin flip. |
| **Breadth** | How many independent bets you make per year. A concentrated fund with 20 positions has low breadth. A systematic fund trading 2,000 stocks daily has massive breadth. |

**Why this formula matters:** It explains why systematic quant funds can compete
with (and often beat) fundamental stock pickers. A quant fund might have a tiny IC
(barely better than random) but enormous breadth (thousands of trades). A
fundamental PM might have a higher IC on each trade but only makes 30 bets a year.

```
Fundamental PM:  IR = 0.10 x sqrt(30)  = 0.55
Quant fund:      IR = 0.02 x sqrt(5000) = 1.41
```

The quant fund wins not because it's smarter on any single trade, but because
it makes vastly more independent bets. This is why the industry has shifted
toward systematic strategies.

### How LPs Use the IR

When your fund talks to investors, the IR is one of the first things they
calculate. A high IR means consistent alpha. A low IR means the alpha (if any)
comes with stomach-churning variability. LPs with long horizons might tolerate
low IR if they believe in the PM. LPs with short leashes (fund of funds,
pension consultants) need to see IR > 0.5 to justify the fee.

---

## 7. Rolling Beta — Because Nothing Is Constant

Everything above assumes that beta, factor loadings, and alpha are fixed numbers.
They're not. They change constantly as market regimes shift, as PMs adjust
positioning, and as the character of stocks evolves.

**Rolling beta** computes beta over a sliding window (typically 63 trading days,
about 3 months) and plots it over time. This reveals how a stock's market
sensitivity evolves.

### What You See in Practice

A stock like AAPL might show:

- **Beta ~0.9 in calm markets** — slightly less volatile than SPY, big stable company
- **Beta ~1.4 during a selloff** — suddenly it's leading the market down, correlations
  spike in a crisis
- **Beta ~0.7 during a rotation into value** — when money flows out of growth,
  AAPL's beta relationship to SPY weakens

This is why your risk desk doesn't just look at a single beta number. A static
beta of 1.0 might mask the fact that beta was 0.7 for ten months and 1.8 for
two months. The average looks fine, but those two months of elevated beta during
a drawdown caused real damage.

### Why Your Desk Monitors Beta Drift

If a PM has a mandate to run at 0.5 net beta and their rolling beta drifts to 1.2,
they're taking on far more market risk than intended. This drift can happen
gradually — a PM adds a few high-beta names, trims some defensive positions, and
before they realize it, they're running a directional book.

Rolling beta catches this early. When you see it in your risk report, that's
the signal to flag it to the PM before it becomes a problem.

### Regime Changes

The most dangerous moment for rolling beta is a regime change — when market
conditions shift abruptly. During COVID in March 2020, correlations spiked
across almost every asset class. Stocks that historically had low beta to the
market suddenly moved in lockstep. Rolling beta charts from that period show
nearly every stock's beta converging toward (or above) 1.0.

This is the scenario that breaks portfolios built on static assumptions.
A "diversified" book with stocks at different beta levels suddenly behaves
like a single levered market bet.

> **Try it yourself:** Run `factor_dashboard.py` and look at rolling beta
> charts for AAPL, JPM, and XOM across 2018-2024. Notice how differently
> they behave during calm periods vs. the COVID selloff.

---

## 8. How to Read the Factor Dashboard Output

When you run the factor model code (see `modules/module_3_factor_models.py`),
you'll get output that looks something like this:

```
  Multi-Factor Model: AAPL
  --------------------------------------------------
    Alpha (annual) :    8.42%
    R-squared      :   48.3%
    Adj R-squared  :   47.9%
  Factor Loadings:
    MKT    :    1.218  (t=35.12) *
    SMB    :   -0.284  (t=-5.67) *
    HML    :   -0.412  (t=-7.89) *

    Information Ratio: 0.61
```

Here's how to read each line:

**Alpha (annual): 8.42%** — After removing factor exposure, AAPL earned 8.42%
per year beyond what the factors explain. But is it statistically significant?
You'd need to check the alpha t-stat (not shown here — look for it in the full
model summary). If the t-stat is below 2, this could be noise.

**R-squared: 48.3%** — Factors explain about half of AAPL's return variation.
The other half is stock-specific. This is reasonable for a mega-cap tech stock —
it moves with the market but also has plenty of idiosyncratic drivers (iPhone
cycles, services growth, regulatory risk).

**Factor Loadings:**

- **MKT = 1.218 (t=35.12)**: AAPL moves about 1.2% for every 1% move in SPY.
  The t-stat of 35 means this is overwhelmingly significant — no question AAPL
  has market exposure.
- **SMB = -0.284 (t=-5.67)**: Negative loading on size. AAPL behaves like a
  large-cap stock (it is one). When small-caps outperform large-caps, AAPL
  tends to lag. Significant.
- **HML = -0.412 (t=-7.89)**: Negative loading on value. AAPL behaves like a
  growth stock. When value outperforms growth, AAPL tends to underperform.
  Significant.

**Information Ratio: 0.61** — For every unit of active risk (tracking error),
AAPL generates 0.61 units of alpha. Above 0.5, which is quite good. But
remember: this is for the stock itself, not for a PM's decision to hold it.

### What to Look For: Red Flags

| Signal | What it might mean |
|--------|--------------------|
| Alpha t-stat < 2 | "Alpha" is probably noise, not skill |
| R^2 > 90% | Closet indexing — paying active fees for passive exposure |
| Negative alpha, significant | Actively destroying value after accounting for factor risk |
| Beta drifting sharply | PM is taking unintended directional risk |
| Unexpected factor exposure | E.g., a "growth" fund with positive HML loading — they're actually buying value stocks |
| Unstable rolling beta | Portfolio character changes with market conditions — hard to risk-manage |

### What to Look For: Good Signs

| Signal | What it might mean |
|--------|--------------------|
| Alpha t-stat > 2 | Genuine stock-picking skill (or at least a very long lucky streak) |
| R^2 in 40-70% range | Active bets with meaningful factor awareness |
| IR > 0.5 | Consistent alpha relative to tracking error |
| Stable rolling beta | PM maintains discipline, factor exposure is intentional |

---

## What to Explore Next

**In this repo:**
- `modules/module_3_factor_models.py` — the Python implementation of everything
  in this guide. Run it and study the output.
- `modules/module_4_portfolio_optimisation.py` — the natural next step: once
  you can decompose returns into factors, how do you *construct* a portfolio
  that maximizes alpha while controlling factor exposure?

**In Paleologo's book:**
- Chapters 6-10 connect factor models to portfolio construction decisions.
  Pay special attention to how he distinguishes between risk factors (things
  you want to hedge) and alpha factors (things you want to bet on).

**Further reading:**
- Grinold & Kahn, *Active Portfolio Management* — the definitive reference on
  information ratio and the fundamental law. Dense but worth it.
- Andrew Ang, *Asset Management* — excellent bridge between academic factor
  theory and how practitioners actually use it.
