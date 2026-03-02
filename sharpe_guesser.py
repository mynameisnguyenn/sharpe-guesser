"""
Sharpe Guesser — Train Your Quantitative Intuition
====================================================

This game shows you a real equity return stream and asks you to guess
the annualised Sharpe ratio. It's the same skill PMs use when eyeballing
a PnL curve and saying "that looks like a Sharpe of 1.2."

The better your intuition here, the faster you can evaluate strategies,
managers, and risk reports without reaching for a spreadsheet.

Usage:
    python sharpe_guesser.py              # default: 5 rounds
    python sharpe_guesser.py --rounds 10  # custom number of rounds
"""

import argparse
import random
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works headless too
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import fetch_prices

TRADING_DAYS = 252

# Tickers to draw from — mix of equities, ETFs, and bonds so you see
# different volatility regimes.
TICKER_POOL = [
    "SPY", "QQQ", "IWM", "DIA",       # indices
    "AAPL", "MSFT", "GOOGL", "AMZN",   # mega-cap tech
    "JPM", "GS", "BAC",                 # financials
    "XOM", "CVX",                        # energy
    "TLT", "IEF", "SHY",               # bonds
    "GLD", "SLV",                        # commodities
    "XLF", "XLE", "XLK", "XLV",        # sectors
]

# Random date windows to keep things interesting
DATE_WINDOWS = [
    ("2015-01-01", "2016-12-31"),
    ("2017-01-01", "2018-12-31"),
    ("2018-06-01", "2020-06-30"),
    ("2019-01-01", "2021-12-31"),
    ("2020-01-01", "2022-12-31"),
    ("2021-01-01", "2023-12-31"),
    ("2022-01-01", "2024-12-31"),
]


def compute_sharpe(daily_returns: pd.Series, annual_rf: float = 0.05) -> float:
    """Annualised Sharpe ratio."""
    daily_rf = annual_rf / TRADING_DAYS
    excess = daily_returns - daily_rf
    ann_ret = excess.mean() * TRADING_DAYS
    ann_vol = excess.std() * np.sqrt(TRADING_DAYS)
    if ann_vol == 0:
        return 0.0
    return ann_ret / ann_vol


def fetch_random_challenge(annual_rf: float = 0.05):
    """
    Pick a random ticker + date window, fetch data, and return
    the info needed for one round of the game.
    """
    random.shuffle(TICKER_POOL)
    start, end = random.choice(DATE_WINDOWS)

    for ticker in TICKER_POOL:
        try:
            prices = fetch_prices(ticker, start, end)
            if len(prices) < 100:
                continue
            rets = prices.pct_change().dropna()
            sr = compute_sharpe(rets, annual_rf)
            if np.isnan(sr) or np.isinf(sr):
                continue
            return {
                "ticker": ticker,
                "start": start,
                "end": end,
                "prices": prices,
                "returns": rets,
                "sharpe": round(sr, 2),
            }
        except Exception:
            continue
    return None


def plot_challenge(challenge: dict, round_num: int) -> str:
    """
    Plot the cumulative return chart (without revealing the ticker)
    and save to a file. Returns the filename.
    """
    rets = challenge["returns"]
    cum = (1 + rets).cumprod()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(cum, linewidth=0.9, color="steelblue")
    ax1.set_title(f"Round {round_num} — Mystery Asset (guess the Sharpe!)")
    ax1.set_ylabel("Growth of $1")
    ax1.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")

    ax2.bar(rets.index, rets.values, width=1, color="steelblue", alpha=0.5)
    ax2.set_ylabel("Daily Return")
    ax2.axhline(0, color="grey", linewidth=0.5)

    plt.tight_layout()
    filename = f"round_{round_num}.png"
    plt.savefig(filename, dpi=120)
    plt.close()
    return filename


def score_guess(guess: float, actual: float) -> dict:
    """Score a guess — closer is better."""
    error = abs(guess - actual)
    if error < 0.1:
        verdict = "Excellent — PM-level intuition"
        points = 3
    elif error < 0.3:
        verdict = "Good — solid risk sense"
        points = 2
    elif error < 0.5:
        verdict = "Decent — keep practising"
        points = 1
    else:
        verdict = "Off — review the chart patterns"
        points = 0
    return {"error": round(error, 2), "verdict": verdict, "points": points}


def run_game(rounds: int = 5, annual_rf: float = 0.05):
    """Main game loop."""
    print("\n" + "=" * 60)
    print("  SHARPE GUESSER — Train Your Quant Intuition")
    print("=" * 60)
    print()
    print("  You'll see a cumulative return chart for a real asset.")
    print("  Your job: guess the annualised Sharpe ratio.")
    print()
    print("  Scoring:")
    print("    Within 0.1  →  3 pts (Excellent)")
    print("    Within 0.3  →  2 pts (Good)")
    print("    Within 0.5  →  1 pt  (Decent)")
    print("    Outside 0.5 →  0 pts")
    print()
    print("  Hints:")
    print("    - A flat line near $1 with no vol → Sharpe near 0")
    print("    - Steady uptrend, low vol → high Sharpe (1.5+)")
    print("    - Wild swings, no clear direction → Sharpe near 0")
    print("    - Steady downtrend → negative Sharpe")
    print()

    total_points = 0
    max_points = rounds * 3
    results = []

    for i in range(1, rounds + 1):
        print(f"\n--- Round {i}/{rounds} ---")
        print("Fetching data...", end=" ", flush=True)

        challenge = fetch_random_challenge(annual_rf)
        if challenge is None:
            print("Could not fetch data. Skipping round.")
            continue

        filename = plot_challenge(challenge, i)
        print(f"Done! Chart saved to: {filename}")
        print("Open the chart, study it, then enter your guess.\n")

        while True:
            try:
                raw = input("  Your Sharpe guess (e.g., 0.8 or -0.3): ").strip()
                if raw.lower() in ("q", "quit", "exit"):
                    print("\nThanks for playing!")
                    sys.exit(0)
                guess = float(raw)
                break
            except ValueError:
                print("  Please enter a number (e.g., 1.2) or 'q' to quit.")

        result = score_guess(guess, challenge["sharpe"])
        total_points += result["points"]

        print(f"\n  Your guess  : {guess:>6.2f}")
        print(f"  Actual      : {challenge['sharpe']:>6.2f}")
        print(f"  Error       : {result['error']:>6.2f}")
        print(f"  Verdict     : {result['verdict']}")
        print(f"  Asset       : {challenge['ticker']}  ({challenge['start']} to {challenge['end']})")
        print(f"  Points      : {result['points']}/3")

        results.append({
            "round": i,
            "ticker": challenge["ticker"],
            "actual": challenge["sharpe"],
            "guess": guess,
            "error": result["error"],
            "points": result["points"],
        })

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SCORE: {total_points}/{max_points}")
    print(f"{'='*60}\n")

    if total_points == max_points:
        print("  Perfect score. You should be running a quant desk.\n")
    elif total_points >= max_points * 0.7:
        print("  Strong showing. Your risk intuition is sharp.\n")
    elif total_points >= max_points * 0.4:
        print("  Not bad. Run this a few more times and you'll improve.\n")
    else:
        print("  Keep at it. Study how vol and trend relate to Sharpe.\n")

    # Print results table
    print(f"  {'Round':<6} {'Ticker':<8} {'Actual':>7} {'Guess':>7} {'Error':>7} {'Pts':>4}")
    print(f"  {'-'*42}")
    for r in results:
        print(
            f"  {r['round']:<6} {r['ticker']:<8} {r['actual']:>7.2f} "
            f"{r['guess']:>7.2f} {r['error']:>7.2f} {r['points']:>4}"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sharpe Guesser — Train your quant intuition")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds (default: 5)")
    parser.add_argument("--rf", type=float, default=0.05, help="Annual risk-free rate (default: 0.05)")
    args = parser.parse_args()
    run_game(rounds=args.rounds, annual_rf=args.rf)
