"""Tests for quiz.spaced_repetition — spaced repetition engine."""

import json
from datetime import datetime, timedelta

import pytest

from quiz.spaced_repetition import (
    compute_streak,
    get_next_questions,
    get_session_summary,
    get_topic_stats,
    get_weak_topics,
    load_progress,
    load_questions,
    record_answer,
    save_progress,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_questions():
    return [
        {"id": "stats_001", "topic": "statistics", "subtopic": "mean", "difficulty": 1,
         "type": "multiple_choice", "source": "module_1",
         "question": "Test question 1", "choices": ["A", "B", "C", "D"],
         "correct": 0, "explanation": "Explanation 1", "tags": ["mean"]},
        {"id": "risk_001", "topic": "risk_metrics", "subtopic": "var", "difficulty": 1,
         "type": "multiple_choice", "source": "module_2",
         "question": "Test question 2", "choices": ["A", "B", "C", "D"],
         "correct": 1, "explanation": "Explanation 2", "tags": ["var"]},
        {"id": "risk_002", "topic": "risk_metrics", "subtopic": "cvar", "difficulty": 2,
         "type": "true_false", "source": "module_2",
         "question": "Test question 3", "choices": ["False", "True"],
         "correct": 1, "explanation": "Explanation 3", "tags": ["cvar"]},
        {"id": "factor_001", "topic": "factor_models", "subtopic": "beta", "difficulty": 1,
         "type": "explain", "source": "module_3",
         "question": "Explain beta", "choices": None, "correct": None,
         "explanation": "Explanation 4", "rubric": "Key points...", "tags": ["beta"]},
        {"id": "stats_002", "topic": "statistics", "subtopic": "vol", "difficulty": 2,
         "type": "multiple_choice", "source": "module_1",
         "question": "Test question 5", "choices": ["A", "B", "C", "D"],
         "correct": 2, "explanation": "Explanation 5", "tags": ["volatility"]},
    ]


@pytest.fixture
def empty_progress():
    return {"cards": {}, "sessions": []}


# ---------------------------------------------------------------------------
# load / save (3 tests)
# ---------------------------------------------------------------------------

def test_load_questions_returns_list(tmp_path, sample_questions):
    """load_questions parses a JSON file and returns a list of question dicts."""
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"questions": sample_questions}), encoding="utf-8")

    result = load_questions(path)
    assert isinstance(result, list)
    assert len(result) == 5
    assert result[0]["id"] == "stats_001"


def test_load_progress_missing_file(tmp_path):
    """load_progress returns empty structure when file does not exist."""
    path = tmp_path / "nonexistent.json"
    result = load_progress(path)

    assert result == {"cards": {}, "sessions": []}


def test_save_and_load_progress_roundtrip(tmp_path):
    """save_progress then load_progress produces identical data."""
    progress = {
        "cards": {
            "stats_001": {
                "streak": 2,
                "ease_factor": 2.6,
                "interval_days": 3,
                "last_reviewed": "2026-03-10T10:00:00",
                "next_review": "2026-03-13T10:00:00",
                "total_attempts": 3,
                "total_correct": 2,
            }
        },
        "sessions": [{"date": "2026-03-10T10:00:00", "total": 5, "correct": 3}],
    }
    path = tmp_path / "progress.json"
    save_progress(progress, path)
    loaded = load_progress(path)

    assert loaded["cards"]["stats_001"]["streak"] == 2
    assert loaded["cards"]["stats_001"]["ease_factor"] == pytest.approx(2.6)
    assert len(loaded["sessions"]) == 1


# ---------------------------------------------------------------------------
# record_answer (6 tests)
# ---------------------------------------------------------------------------

def test_record_answer_correct_increments_streak(empty_progress):
    """Correct answer increments streak from 0 to 1."""
    card = record_answer(empty_progress, "stats_001", correct=True)
    assert card["streak"] == 1


def test_record_answer_correct_increases_ease(empty_progress):
    """Correct answer increases ease_factor by 0.1."""
    original_ease = 2.5  # DEFAULT_EASE_FACTOR
    card = record_answer(empty_progress, "stats_001", correct=True)
    assert card["ease_factor"] == pytest.approx(original_ease + 0.1)


def test_record_answer_incorrect_resets_streak(empty_progress):
    """Incorrect answer resets streak to 0."""
    # Build up a streak first
    record_answer(empty_progress, "stats_001", correct=True)
    record_answer(empty_progress, "stats_001", correct=True)
    card = record_answer(empty_progress, "stats_001", correct=False)
    assert card["streak"] == 0


def test_record_answer_incorrect_decreases_ease(empty_progress):
    """Incorrect answer decreases ease_factor by 0.2."""
    original_ease = 2.5
    card = record_answer(empty_progress, "stats_001", correct=False)
    assert card["ease_factor"] == pytest.approx(original_ease - 0.2)


def test_record_answer_ease_floor(empty_progress):
    """Ease factor never drops below 1.3."""
    # Drive ease down with many wrong answers
    for _ in range(20):
        card = record_answer(empty_progress, "stats_001", correct=False)
    assert card["ease_factor"] == pytest.approx(1.3)


def test_record_answer_self_rating_overrides(empty_progress):
    """self_rating >= 3 treated as correct regardless of 'correct' flag."""
    # Pass correct=False but self_rating=4 -> should count as correct
    card = record_answer(empty_progress, "factor_001", correct=False, self_rating=4)
    assert card["streak"] == 1
    assert card["total_correct"] == 1

    # self_rating=2 -> should count as incorrect
    card = record_answer(empty_progress, "factor_001", correct=True, self_rating=2)
    assert card["streak"] == 0


# ---------------------------------------------------------------------------
# get_next_questions (6 tests)
# ---------------------------------------------------------------------------

def test_get_next_questions_unseen_first(sample_questions, empty_progress):
    """Unseen questions come before not-yet-due questions."""
    # Mark one question as seen and not due (future next_review)
    future = (datetime.now() + timedelta(days=5)).isoformat()
    empty_progress["cards"]["stats_001"] = {
        "streak": 1, "ease_factor": 2.6, "interval_days": 3,
        "last_reviewed": datetime.now().isoformat(),
        "next_review": future,
        "total_attempts": 1, "total_correct": 1,
    }

    result = get_next_questions(empty_progress, sample_questions, n=5)
    ids = [q["id"] for q in result]

    # stats_001 is not due yet; unseen questions should appear before it
    assert ids.index("risk_001") < ids.index("stats_001")


def test_get_next_questions_overdue_before_unseen(sample_questions, empty_progress):
    """Overdue questions have highest priority over unseen."""
    # Mark one question as overdue (past next_review)
    past = (datetime.now() - timedelta(days=3)).isoformat()
    empty_progress["cards"]["risk_001"] = {
        "streak": 1, "ease_factor": 2.5, "interval_days": 1,
        "last_reviewed": (datetime.now() - timedelta(days=4)).isoformat(),
        "next_review": past,
        "total_attempts": 1, "total_correct": 1,
    }

    result = get_next_questions(empty_progress, sample_questions, n=5)
    ids = [q["id"] for q in result]

    # Overdue risk_001 should come first
    assert ids[0] == "risk_001"


def test_get_next_questions_respects_n(sample_questions, empty_progress):
    """Returns at most n questions."""
    result = get_next_questions(empty_progress, sample_questions, n=2)
    assert len(result) == 2


def test_get_next_questions_topic_filter(sample_questions, empty_progress):
    """mode='topic' only returns matching topic when topic is set."""
    result = get_next_questions(
        empty_progress, sample_questions, n=10, mode="topic", topic="risk_metrics"
    )
    assert len(result) == 2
    assert all(q["topic"] == "risk_metrics" for q in result)


def test_get_next_questions_random_mode(sample_questions, empty_progress):
    """mode='random' returns n questions."""
    result = get_next_questions(empty_progress, sample_questions, n=3, mode="random")
    assert len(result) == 3


def test_get_next_questions_no_duplicates(sample_questions, empty_progress):
    """No duplicate questions in result."""
    result = get_next_questions(empty_progress, sample_questions, n=5)
    ids = [q["id"] for q in result]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# get_topic_stats (3 tests)
# ---------------------------------------------------------------------------

def test_get_topic_stats_empty_progress(empty_progress):
    """Returns empty dict when no cards exist."""
    stats = get_topic_stats(empty_progress)
    assert stats == {}


def test_get_topic_stats_computes_accuracy(empty_progress):
    """Accuracy is computed as correct / attempts."""
    record_answer(empty_progress, "risk_001", correct=True)
    record_answer(empty_progress, "risk_001", correct=False)

    stats = get_topic_stats(empty_progress)
    assert "risk" in stats
    assert stats["risk"]["attempts"] == 2
    assert stats["risk"]["correct"] == 1
    assert stats["risk"]["accuracy"] == pytest.approx(0.5)


def test_get_topic_stats_identifies_mastered(empty_progress):
    """A card with streak >= 3 and ease_factor >= 2.5 counts as mastered."""
    # Answer correctly 3 times -> streak=3, ease=2.5+0.3=2.8
    for _ in range(3):
        record_answer(empty_progress, "stats_001", correct=True)

    stats = get_topic_stats(empty_progress)
    assert stats["stats"]["mastered"] == 1


# ---------------------------------------------------------------------------
# get_weak_topics (2 tests)
# ---------------------------------------------------------------------------

def test_get_weak_topics_below_threshold(empty_progress):
    """Returns topics with accuracy below 0.6."""
    # risk: 1 correct out of 4 = 25% accuracy
    record_answer(empty_progress, "risk_001", correct=True)
    record_answer(empty_progress, "risk_001", correct=False)
    record_answer(empty_progress, "risk_001", correct=False)
    record_answer(empty_progress, "risk_001", correct=False)

    weak = get_weak_topics(empty_progress)
    assert "risk" in weak


def test_get_weak_topics_empty_when_all_strong(empty_progress):
    """Returns [] when all topics are above threshold."""
    # stats: 3 correct out of 3 = 100%
    for _ in range(3):
        record_answer(empty_progress, "stats_001", correct=True)

    weak = get_weak_topics(empty_progress)
    assert weak == []


# ---------------------------------------------------------------------------
# compute_streak (3 tests)
# ---------------------------------------------------------------------------

def test_compute_streak_no_sessions(empty_progress):
    """Returns 0 when no sessions exist."""
    assert compute_streak(empty_progress) == 0


def test_compute_streak_consecutive_days():
    """Counts consecutive days of activity."""
    today = datetime.now()
    progress = {
        "cards": {},
        "sessions": [
            {"date": (today - timedelta(days=2)).isoformat(), "total": 5, "correct": 3},
            {"date": (today - timedelta(days=1)).isoformat(), "total": 5, "correct": 4},
            {"date": today.isoformat(), "total": 5, "correct": 5},
        ],
    }
    assert compute_streak(progress) == 3


def test_compute_streak_gap_breaks():
    """A gap in dates resets the streak."""
    today = datetime.now()
    progress = {
        "cards": {},
        "sessions": [
            {"date": (today - timedelta(days=5)).isoformat(), "total": 5, "correct": 3},
            # gap: days 4, 3, 2 missing
            {"date": (today - timedelta(days=1)).isoformat(), "total": 5, "correct": 4},
            {"date": today.isoformat(), "total": 5, "correct": 5},
        ],
    }
    # Only today and yesterday are consecutive
    assert compute_streak(progress) == 2


# ---------------------------------------------------------------------------
# get_session_summary (2 tests)
# ---------------------------------------------------------------------------

def test_get_session_summary_basic(empty_progress):
    """Correct totals and accuracy in session summary."""
    session_answers = [
        {"question_id": "stats_001", "correct": True, "topic": "statistics"},
        {"question_id": "risk_001", "correct": False, "topic": "risk_metrics"},
        {"question_id": "risk_002", "correct": True, "topic": "risk_metrics"},
    ]

    summary = get_session_summary(empty_progress, session_answers)
    assert summary["total"] == 3
    assert summary["correct"] == 2
    assert summary["accuracy"] == pytest.approx(2 / 3)


def test_get_session_summary_per_topic(empty_progress):
    """Breakdown by topic is computed correctly."""
    session_answers = [
        {"question_id": "stats_001", "correct": True, "topic": "statistics"},
        {"question_id": "stats_002", "correct": False, "topic": "statistics"},
        {"question_id": "risk_001", "correct": True, "topic": "risk_metrics"},
    ]

    summary = get_session_summary(empty_progress, session_answers)
    per_topic = summary["per_topic"]

    assert per_topic["statistics"]["total"] == 2
    assert per_topic["statistics"]["correct"] == 1
    assert per_topic["statistics"]["accuracy"] == pytest.approx(0.5)

    assert per_topic["risk_metrics"]["total"] == 1
    assert per_topic["risk_metrics"]["correct"] == 1
    assert per_topic["risk_metrics"]["accuracy"] == pytest.approx(1.0)
