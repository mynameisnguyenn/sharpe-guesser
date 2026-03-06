# Quant Finance Bootcamp — A Practical Roadmap

You're a risk & data associate at a tiger cub. You have market intuition,
exposure to PMs and quants, and access to real PnL data. That's a huge head
start. Most quant aspirants have the math but not the market context. You
have the context — now you need the tools.

---

## Months 1–2: Python Fundamentals Through Finance

**Goal:** Get comfortable writing Python that does real financial computation.

- Learn Python basics through modules 1–2 in this repo
- **Project:** Rebuild an Excel report in Python (`sharpe_101.py`, `sharpe_guesser.py`)
- **Resource:** Paleologo, *Advanced Portfolio Management* — chapters 1–5
- **Skills:** variables, functions, pandas, numpy, matplotlib, APIs

**Exercises:**
1. Run `sharpe_101.py` and modify it to analyse your fund's benchmark
2. Play `sharpe_guesser.py` until you can guess within 0.3 consistently
3. Add a new function to `sharpe_101.py` that computes the Sortino ratio

---

## Months 3–4: Risk Metrics and Factor Models

**Goal:** Automate the reports you already read manually.

- Work through modules 2–3
- Learn pandas deeply — it's 60% of the job
- **Project:** Factor exposure dashboard (`factor_dashboard.py` in this repo)
- **Resource:** Paleologo, *Advanced Portfolio Management* — chapters 6–10
- **Skills:** scipy, statsmodels, regression, risk metrics

**Why this matters:** These are the reports your desk produces. Building them
from scratch means you understand what the numbers actually mean, not just
what cell to look at.

---

## Months 5–6: Portfolio Construction

**Goal:** Go from "measuring risk" to "constructing portfolios."

- Work through modules 4–5
- Learn `scipy.optimize`
- **Project:** Efficient frontier tool with backtest comparing optimized vs equal-weight
- **Resource:** Grinold & Kahn, *Active Portfolio Management* (info ratio, fundamental law)
- **Skills:** optimization, backtesting, strategy evaluation

---

## Months 7–9: Build Something Real at Work

This is the most important phase.

- Automate a risk report, build a screening tool, or create a dashboard
- Ship code people at your fund actually use
- This is worth more than any certificate

---

## Months 10–12: Specialize and Apply

At this point you'll know which direction fits:

**Path A: Quant Risk** (natural extension)
- Stress testing frameworks
- Greeks and options pricing (Black-Scholes, Monte Carlo)
- Regulatory capital (Basel III/IV)

**Path B: Quantitative PM / Researcher** (bigger leap)
- Signal research and systematic strategies
- Factor investing, momentum, mean reversion
- Machine learning for alpha (tree models first, not deep learning)

Start applying after month 6. Your desk experience + GitHub portfolio is a
real differentiator.

---

## What NOT to Waste Time On

- **Don't get a masters yet** — build things first
- **Don't grind LeetCode** — that's for SWE, not quant risk/research
- **Don't learn C++** — Python is enough for target roles
- **Don't buy courses** — build projects with your market intuition
- **Don't wait until you feel "ready"**

---

## The Real Edge

You sit next to PMs and quants every day. Ask them:

- "What tool do you wish existed?"
- "What's still manual?"

Build that tool. That's your resume.

---

## Recommended Resources

**Books (in order):**
1. *Advanced Portfolio Management* — Marco Paleologo (practical, directly relevant)
2. *Python for Finance* — Yves Hilpisch (code-heavy, good reference)
3. *Active Portfolio Management* — Grinold & Kahn (the classic on alpha and IC)
4. *Advances in Financial Machine Learning* — Marcos Lopez de Prado (the modern quant bible)
5. *Quantitative Risk Management* — McNeil, Frey, Embrechts (your risk background makes this accessible)

**Practice:**
- This repo — keep adding to it
- Kaggle financial datasets
- QuantConnect / Zipline for backtesting
