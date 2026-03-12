# CLAUDE.md — Quant Finance Quiz

## What this is
A spaced repetition quiz system for quant finance concepts. Two interfaces:
1. **Claude Code `/quiz` slash command** — conversational quiz on phone/commute
2. **Streamlit dashboard** — progress tracking and study reference at home/office

Both share the same question bank (`questions.json`) and progress file (`progress.json`).

## How to run
```bash
# Streamlit dashboard
streamlit run quiz/app.py

# Claude Code slash command
/quiz              # 10 questions, smart mode
/quiz 5            # 5 questions
/quiz topic factor_models  # drill one topic
/quiz weak         # only weak areas
/quiz random       # random, no scheduling
```

## File structure
- `spaced_repetition.py` — SR algorithm + progress tracking (pure Python, no Streamlit)
- `questions.json` — ~200 curated questions across 9 topics, 3 difficulty levels
- `progress.json` — user progress (gitignored, auto-created)
- `app.py` — Streamlit dashboard (3 tabs: Quiz, Progress, Browse)

## 9 topics
| Topic | Source |
|-------|--------|
| statistics | Module 1 |
| risk_metrics | Module 2, Paleologo Ch 3 |
| factor_models | Module 3, Paleologo Ch 4-5 |
| portfolio_opt | Module 4, Paleologo Ch 6 |
| strategies | Module 5 |
| alpha_sizing | Paleologo Ch 6 |
| factor_risk | Paleologo Ch 7-8 |
| risk_mgmt | Paleologo Ch 9-10 |
| projects | Kelly ML, Vol Forecasting, Regime Detection |

## Spaced repetition algorithm
- Correct: streak += 1, interval grows (0→1→3→prev*ease), ease += 0.1
- Incorrect: streak = 0, interval = 0 (immediate), ease -= 0.2 (min 1.3)
- Selection priority: overdue > weak unseen > unseen > not-due

## Tests
```bash
python -m pytest tests/test_quiz.py -v
```

## Dependencies
No new dependencies. Uses streamlit (already installed).
