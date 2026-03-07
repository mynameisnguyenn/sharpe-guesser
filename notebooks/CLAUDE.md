# CLAUDE.md — Notebooks

## What these are
Interactive Jupyter notebooks for hands-on learning. The user opens these in VS Code
and runs cells to build intuition about factor model math.

## Notebooks
- `factor_models_explained.ipynb` — Factor models with **real market data** (AAPL, JPM, XOM).
  32 cells covering beta, CAPM, Fama-French, rolling beta, Information Ratio.
  Downloads data via yfinance when run.

- `factor_math_interactive.ipynb` — Interactive version of FACTOR_MATH_GUIDE.md with
  **synthetic data**. 22 cells. Cells marked "TRY IT" have variables users can change.
  No network calls needed — all data is generated with numpy.

## Conventions
- Use `%matplotlib inline` (not Agg backend — notebooks should show charts inline)
- Use `plt.show()` in notebooks (they render inline, not in separate windows)
- Kernel: "sharpe-guesser" (registered via ipykernel from the project venv)
- When creating new notebooks, prefer writing raw .ipynb JSON via Write tool
  (the NotebookEdit tool has crashed previously)
- Keep markdown cells concise — hedge fund tone, practical, no academic fluff
- Include exercises as separate cells with commented-out starter code

## Jupyter kernel setup
```bash
pip install ipykernel
python -m ipykernel install --user --name sharpe-guesser --display-name "sharpe-guesser"
```
