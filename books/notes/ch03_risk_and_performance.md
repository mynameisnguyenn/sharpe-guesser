# Chapter 3: A Tour of Risk and Performance

## Alpha and Beta

**Beta** = sensitivity of a stock to the market.
- Regression of stock returns on market (SPY) returns.
- SYF (cyclical financial) has high beta; WMT (defensive) has low beta.
- Beta tells you: "if market returns 1%, this stock returns beta%."

**Alpha** = the return left over after removing the market's contribution.
- `stock_return = alpha + beta * market_return + epsilon`
- Alpha is what a PM gets paid for. Beta is what an index fund delivers.

## The Single-Factor Model
```
r_stock = alpha + beta * r_market + epsilon
```
- `epsilon` = idiosyncratic (stock-specific) return
- Key property: epsilon is uncorrelated with the market and across stocks
- This is CAPM in its simplest form

## Risk Decomposition (Pythagoras)

**The core insight**: for independent (uncorrelated) components, variances add:
```
Var(total) = Var(factor) + Var(idio)
```

This is why the book uses *variance* (not volatility) for decomposition — variance
is additive, volatility is not.

**Example**: If a portfolio has 15% total annual vol, and the factor component
has 5% vol, then:
- Factor variance = 25 (= 5^2)
- Total variance = 225 (= 15^2)
- Idio variance = 200 (= 225 - 25)
- Idio vol = sqrt(200) = 14.1%
- % idio var = 200/225 = 89%

## Performance Attribution
Break total PnL into:
- **Factor PnL** = sum of (factor exposure * factor return) for each factor
- **Idiosyncratic PnL** = total PnL - factor PnL

For a fundamental PM, idio PnL is the only thing that reflects skill.
Factor PnL is "riding the market" — any index fund can do that.

## Simple Hedging
To zero out market beta:
1. Compute portfolio's dollar beta exposure = sum of (beta_i * NMV_i)
2. Short that dollar amount of SPY (or S&P futures)
3. Now the portfolio is market-neutral — its returns are purely idiosyncratic

This is the simplest form of "separation of concerns."

## Volatility Scaling
- Annual vol = daily vol * sqrt(252)
- Weekly vol = annual vol / sqrt(52)
- Monthly vol = annual vol / sqrt(12)
- Average absolute daily return ≈ 0.8 * daily vol (for normal distribution)

## Key insight: You can't estimate expected returns from more frequent data
- Splitting a year into n intervals: expected return per period = alpha/n
- Estimation error per period = sigma/n
- Signal-to-noise ratio = alpha/sigma — independent of n!
- More data points don't help if the signal is faint
