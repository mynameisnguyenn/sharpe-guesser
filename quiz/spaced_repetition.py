"""
Spaced Repetition Engine for Quant Finance Quiz
================================================

Pure logic module — no Streamlit dependency. Handles question loading,
progress tracking, and SM-2-inspired spaced repetition scheduling.

Public API:
    load_questions(path) -> list[dict]
    load_progress(path) -> dict
    save_progress(progress, path)
    get_next_questions(progress, questions, n, mode, topic) -> list[dict]
    record_answer(progress, question_id, correct, self_rating) -> dict
    get_topic_stats(progress) -> dict
    get_weak_topics(progress, threshold) -> list[str]
    compute_streak(progress) -> int
    get_session_summary(progress, session_answers) -> dict
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

# Defaults
DEFAULT_EASE_FACTOR = 2.5
MIN_EASE_FACTOR = 1.3
INTERVALS = [0, 1, 3]  # day 0 (immediate), 1 day, 3 days, then *= ease


def load_questions(path: str | Path) -> list[dict]:
    """Load questions from JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"] if "questions" in data else data


def load_progress(path: str | Path) -> dict:
    """Load progress from JSON file. Returns empty structure if file missing."""
    path = Path(path)
    if not path.exists():
        return {"cards": {}, "sessions": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(progress: dict, path: str | Path) -> None:
    """Save progress to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, default=str)


def _get_card(progress: dict, question_id: str) -> dict:
    """Get or create a card entry for a question."""
    if question_id not in progress["cards"]:
        progress["cards"][question_id] = {
            "streak": 0,
            "ease_factor": DEFAULT_EASE_FACTOR,
            "interval_days": 0,
            "last_reviewed": None,
            "next_review": None,
            "total_attempts": 0,
            "total_correct": 0,
        }
    return progress["cards"][question_id]


def _is_due(card: dict, now: datetime | None = None) -> bool:
    """Check if a card is due for review."""
    if card["next_review"] is None:
        return True  # never reviewed
    now = now or datetime.now()
    next_dt = datetime.fromisoformat(card["next_review"])
    return now >= next_dt


def _days_overdue(card: dict, now: datetime | None = None) -> float:
    """How many days overdue a card is. Negative = not yet due."""
    if card["next_review"] is None:
        return float("inf")  # unseen cards are maximally overdue
    now = now or datetime.now()
    next_dt = datetime.fromisoformat(card["next_review"])
    return (now - next_dt).total_seconds() / 86400


def record_answer(
    progress: dict,
    question_id: str,
    correct: bool,
    self_rating: int | None = None,
) -> dict:
    """
    Record an answer and update the spaced repetition schedule.

    Parameters
    ----------
    progress : dict
        The progress data structure.
    question_id : str
        ID of the question answered.
    correct : bool
        Whether the answer was correct.
    self_rating : int or None
        For 'explain' questions: 1-5 self-rating. If provided and >= 3,
        treated as correct for scheduling purposes.

    Returns
    -------
    dict
        The updated card for this question.
    """
    card = _get_card(progress, question_id)
    now = datetime.now()

    # For explain questions, self_rating overrides correct
    if self_rating is not None:
        correct = self_rating >= 3

    card["total_attempts"] += 1

    if correct:
        card["total_correct"] += 1
        card["streak"] += 1
        card["ease_factor"] = min(card["ease_factor"] + 0.1, 3.0)

        # Compute next interval
        streak = card["streak"]
        if streak <= len(INTERVALS):
            card["interval_days"] = INTERVALS[streak - 1]
        else:
            prev = card["interval_days"] if card["interval_days"] > 0 else INTERVALS[-1]
            card["interval_days"] = math.ceil(prev * card["ease_factor"])
    else:
        card["streak"] = 0
        card["interval_days"] = 0  # review next session
        card["ease_factor"] = max(card["ease_factor"] - 0.2, MIN_EASE_FACTOR)

    card["last_reviewed"] = now.isoformat()
    card["next_review"] = (now + timedelta(days=card["interval_days"])).isoformat()

    return card


def get_next_questions(
    progress: dict,
    questions: list[dict],
    n: int = 10,
    mode: str = "smart",
    topic: str | None = None,
) -> list[dict]:
    """
    Select the next batch of questions using spaced repetition.

    Modes
    -----
    smart : Default SR scheduling (overdue > weak > unseen > not-yet-due)
    topic : Filter to a single topic, then apply SR
    random : Random selection (no SR)
    weak : Only questions from weak topics (accuracy < 60%)
    """
    now = datetime.now()
    pool = list(questions)

    # Filter by topic if specified
    if topic is not None:
        pool = [q for q in pool if q["topic"] == topic]
    elif mode == "weak":
        weak = set(get_weak_topics(progress))
        if weak:
            pool = [q for q in pool if q["topic"] in weak]

    if mode == "random":
        import random
        random.shuffle(pool)
        return pool[:n]

    # Categorize questions
    overdue = []
    unseen = []
    not_due = []

    for q in pool:
        qid = q["id"]
        if qid not in progress.get("cards", {}):
            unseen.append(q)
            continue
        card = progress["cards"][qid]
        if _is_due(card, now):
            overdue.append((q, _days_overdue(card, now)))
        else:
            not_due.append((q, _days_overdue(card, now)))

    # Sort overdue by most overdue first
    overdue.sort(key=lambda x: x[1], reverse=True)

    # Sort not-due by soonest due first
    not_due.sort(key=lambda x: x[1], reverse=True)

    # Prioritize weak-topic unseen questions
    weak_topics = set(get_weak_topics(progress))
    unseen_weak = [q for q in unseen if q["topic"] in weak_topics]
    unseen_other = [q for q in unseen if q["topic"] not in weak_topics]

    # Build result: overdue first, then weak unseen, then other unseen, then not-due
    result = []
    for q, _ in overdue:
        if len(result) >= n:
            break
        result.append(q)
    for q in unseen_weak:
        if len(result) >= n:
            break
        result.append(q)
    for q in unseen_other:
        if len(result) >= n:
            break
        result.append(q)
    for q, _ in not_due:
        if len(result) >= n:
            break
        result.append(q)

    return result[:n]


def get_topic_stats(progress: dict, questions: list[dict] | None = None) -> dict:
    """
    Compute per-topic statistics from progress data.

    Parameters
    ----------
    progress : dict
        The progress data structure.
    questions : list[dict] or None
        If provided, used to map question IDs to topics accurately.
        If None, topic is extracted from the ID prefix (best-effort).

    Returns dict of {topic: {attempts, correct, accuracy, mastered, due}}.
    """
    # Build qid -> topic mapping from questions if available
    qid_topic = {}
    if questions:
        for q in questions:
            qid_topic[q["id"]] = q["topic"]

    topic_data = {}

    for qid, card in progress.get("cards", {}).items():
        if qid in qid_topic:
            topic = qid_topic[qid]
        else:
            # Fallback: extract topic from ID prefix (e.g., "risk_metrics_015")
            parts = qid.rsplit("_", 1)
            topic = parts[0] if len(parts) == 2 and parts[1].isdigit() else qid

        if topic not in topic_data:
            topic_data[topic] = {
                "attempts": 0,
                "correct": 0,
                "accuracy": 0.0,
                "mastered": 0,
                "due": 0,
            }

        td = topic_data[topic]
        td["attempts"] += card["total_attempts"]
        td["correct"] += card["total_correct"]

        # Mastered = streak >= 3 and ease_factor >= 2.5
        if card["streak"] >= 3 and card["ease_factor"] >= 2.5:
            td["mastered"] += 1

        # Due for review
        if _is_due(card):
            td["due"] += 1

    # Compute accuracy
    for topic, td in topic_data.items():
        if td["attempts"] > 0:
            td["accuracy"] = td["correct"] / td["attempts"]

    return topic_data


def get_weak_topics(progress: dict, threshold: float = 0.6) -> list[str]:
    """Return topics with accuracy below threshold."""
    stats = get_topic_stats(progress)
    return [
        topic
        for topic, td in stats.items()
        if td["attempts"] > 0 and td["accuracy"] < threshold
    ]


def compute_streak(progress: dict) -> int:
    """
    Compute the current study streak in consecutive days.

    Looks at session timestamps to count consecutive days with activity.
    """
    sessions = progress.get("sessions", [])
    if not sessions:
        return 0

    # Get unique dates with activity, sorted descending
    dates = set()
    for s in sessions:
        if "date" in s:
            try:
                dt = datetime.fromisoformat(s["date"]).date()
                dates.add(dt)
            except (ValueError, TypeError):
                continue

    if not dates:
        return 0

    sorted_dates = sorted(dates, reverse=True)
    today = datetime.now().date()

    # Streak must include today or yesterday
    if sorted_dates[0] < today - timedelta(days=1):
        return 0

    streak = 1
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i] == sorted_dates[i - 1] - timedelta(days=1):
            streak += 1
        else:
            break

    return streak


def get_session_summary(progress: dict, session_answers: list[dict]) -> dict:
    """
    Summarize a quiz session.

    Parameters
    ----------
    progress : dict
        Current progress data.
    session_answers : list[dict]
        List of {"question_id": str, "correct": bool, "topic": str}.

    Returns
    -------
    dict with total, correct, accuracy, per_topic breakdown.
    """
    total = len(session_answers)
    correct = sum(1 for a in session_answers if a["correct"])

    # Per-topic breakdown
    topic_breakdown = {}
    for a in session_answers:
        t = a.get("topic", "unknown")
        if t not in topic_breakdown:
            topic_breakdown[t] = {"total": 0, "correct": 0}
        topic_breakdown[t]["total"] += 1
        if a["correct"]:
            topic_breakdown[t]["correct"] += 1

    for t, tb in topic_breakdown.items():
        tb["accuracy"] = tb["correct"] / tb["total"] if tb["total"] > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "per_topic": topic_breakdown,
        "streak": compute_streak(progress),
    }


def record_session(progress: dict, session_answers: list[dict]) -> None:
    """Record a completed session in progress history."""
    if "sessions" not in progress:
        progress["sessions"] = []
    progress["sessions"].append({
        "date": datetime.now().isoformat(),
        "total": len(session_answers),
        "correct": sum(1 for a in session_answers if a["correct"]),
    })


def get_all_topics(questions: list[dict]) -> list[str]:
    """Get sorted list of unique topics from question bank."""
    return sorted(set(q["topic"] for q in questions))


def get_questions_by_topic(questions: list[dict], topic: str) -> list[dict]:
    """Filter questions by topic."""
    return [q for q in questions if q["topic"] == topic]


def get_due_count(progress: dict) -> int:
    """Count how many cards are currently due for review."""
    now = datetime.now()
    return sum(1 for card in progress.get("cards", {}).values() if _is_due(card, now))


def get_mastery_percentage(progress: dict, total_questions: int) -> float:
    """Percentage of questions mastered (streak >= 3, ease >= 2.5)."""
    if total_questions == 0:
        return 0.0
    mastered = sum(
        1 for card in progress.get("cards", {}).values()
        if card["streak"] >= 3 and card["ease_factor"] >= 2.5
    )
    return mastered / total_questions
