# Quant Finance Quiz — Spaced Repetition Learning Companion

A quiz system that tests understanding of quant finance concepts using spaced repetition scheduling. Two interfaces to the same question bank:

## Interfaces

### Claude Code (`/quiz`)
Conversational quiz on your phone during commute. Ask follow-up questions, debate answers, get deeper explanations.

```
/quiz                        # 10 questions, smart scheduling
/quiz 5                      # 5 questions
/quiz topic risk_metrics     # drill one topic
/quiz weak                   # focus on weak areas
/quiz random                 # random, no scheduling
```

### Streamlit Dashboard
Visual progress tracking and study reference at home/office.

```bash
streamlit run quiz/app.py
```

**Tab 1 — Quiz:** Interactive quiz with immediate feedback and explanations.
**Tab 2 — Progress:** Topic heatmap, accuracy stats, study streak, mastery percentage.
**Tab 3 — Browse:** Scroll all questions + explanations as a study reference.

## Question Bank

~200 questions across 9 topics at 3 difficulty levels:

| Topic | Questions | Source Material |
|-------|-----------|----------------|
| Statistics | 18 | Module 1: distributions, skewness, kurtosis |
| Risk Metrics | 25 | Module 2: VaR, CVaR, drawdowns, Sortino |
| Factor Models | 28 | Module 3, Paleologo Ch 4-5: CAPM, FF3, beta |
| Portfolio Optimization | 22 | Module 4: Markowitz, risk parity |
| Strategies | 20 | Module 5: momentum, mean reversion, backtesting |
| Alpha Sizing | 18 | Paleologo Ch 6: proportional vs MV sizing |
| Factor Risk | 22 | Paleologo Ch 7-8: risk decomposition, hedging |
| Risk Management | 16 | Paleologo Ch 9-10: stop-loss, leverage |
| Projects | 25 | Kelly ML, vol forecasting, regime detection |

**Difficulty levels:**
- Level 1 (Foundation): "What does beta of 1.3 mean?"
- Level 2 (Practitioner): "Your PM's rolling beta jumped — what might explain this?"
- Level 3 (Advanced): Cross-topic synthesis, interview-level questions

## How It Works

The spaced repetition algorithm schedules questions based on your performance:

- **Correct answers** push the question further into the future (increasing intervals)
- **Wrong answers** bring the question back immediately
- **Smart selection** prioritizes: overdue > weak topics > unseen > mastered

Progress is saved to `quiz/progress.json` (gitignored) and persists across sessions.

## Running Tests

```bash
python -m pytest tests/test_quiz.py -v
```
