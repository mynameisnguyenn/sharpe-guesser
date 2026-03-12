"""
Quant Finance Quiz -- Spaced Repetition Dashboard
=================================================
Run: streamlit run quiz/app.py
"""

from pathlib import Path

import streamlit as st

from quiz.spaced_repetition import (
    compute_streak,
    get_all_topics,
    get_due_count,
    get_mastery_percentage,
    get_next_questions,
    get_session_summary,
    get_topic_stats,
    load_progress,
    load_questions,
    record_answer,
    record_session,
    save_progress,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
QUIZ_DIR = Path(__file__).parent
QUESTIONS_PATH = QUIZ_DIR / "questions.json"
PROGRESS_PATH = QUIZ_DIR / "progress.json"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Quant Quiz",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MODE_MAP = {
    "Smart (SR)": "smart",
    "Topic Focus": "topic",
    "Random": "random",
    "Weak Areas": "weak",
}

DIFFICULTY_STARS = {1: "*", 2: "**", 3: "***", 4: "****", 5: "*****"}


def _display_topic(topic: str) -> str:
    """Convert topic slug to display name."""
    return topic.replace("_", " ").title()


def _difficulty_label(level: int) -> str:
    """Return a star-based difficulty indicator."""
    return DIFFICULTY_STARS.get(level, "*" * level)


@st.cache_data
def _load_questions() -> list[dict]:
    """Load question bank (cached -- questions don't change at runtime)."""
    return load_questions(QUESTIONS_PATH)


def _load_progress() -> dict:
    """Load progress (NOT cached -- changes every answer)."""
    return load_progress(PROGRESS_PATH)


def _save(progress: dict) -> None:
    """Persist progress to disk."""
    save_progress(progress, PROGRESS_PATH)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def _init_state() -> None:
    """Ensure all session-state keys exist."""
    defaults = {
        "quiz_active": False,
        "quiz_questions": [],
        "quiz_index": 0,
        "quiz_answers": [],
        "show_explanation": False,
        "last_correct": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ---------------------------------------------------------------------------
# Tab 1: Quiz
# ---------------------------------------------------------------------------
def _render_setup(questions: list[dict], progress: dict) -> None:
    """Render the quiz setup panel (before a quiz starts)."""
    st.subheader("Start a Quiz")

    n_questions = st.slider(
        "Number of questions", min_value=5, max_value=30, value=10
    )

    mode_label = st.selectbox("Mode", list(MODE_MAP.keys()))
    mode = MODE_MAP[mode_label]

    selected_topic = None
    if mode_label == "Topic Focus":
        topics = get_all_topics(questions)
        display_topics = [_display_topic(t) for t in topics]
        choice = st.selectbox("Topic", display_topics)
        selected_topic = topics[display_topics.index(choice)]

    st.divider()

    due = get_due_count(progress)
    if due > 0:
        st.info(f"{due} question(s) due for review.")

    if st.button("Start Quiz", type="primary", use_container_width=True):
        selected = get_next_questions(
            progress, questions, n=n_questions, mode=mode, topic=selected_topic
        )
        if not selected:
            st.warning("No questions match the current filters. Try a different mode.")
            return
        st.session_state.quiz_active = True
        st.session_state.quiz_questions = selected
        st.session_state.quiz_index = 0
        st.session_state.quiz_answers = []
        st.session_state.show_explanation = False
        st.session_state.last_correct = False
        st.rerun()


def _render_question(question: dict, index: int, total: int) -> None:
    """Render a single question card with answer input."""
    # Progress bar
    st.progress((index + 1) / total, text=f"Question {index + 1} of {total}")

    # Topic badge + difficulty
    topic_display = _display_topic(question.get("topic", "general"))
    difficulty = question.get("difficulty", 1)
    st.caption(f"{topic_display}  |  Difficulty: {_difficulty_label(difficulty)}")

    st.divider()

    # Question text
    st.markdown(f"**{question['question']}**")

    q_type = question.get("type", "multiple_choice")

    if q_type == "multiple_choice":
        choices = question.get("choices", [])
        answer = st.radio(
            "Select your answer:",
            choices,
            index=None,
            key=f"mc_answer_{question['id']}",
        )
        if st.button("Submit Answer", type="primary", use_container_width=True):
            if answer is None:
                st.warning("Please select an answer.")
                return
            correct_idx = question.get("correct", 0)
            is_correct = choices.index(answer) == correct_idx
            _process_answer(question, is_correct)

    elif q_type == "true_false":
        choices = question.get("choices", ["False", "True"])
        answer = st.radio(
            "True or False:",
            choices,
            index=None,
            key=f"tf_answer_{question['id']}",
        )
        if st.button("Submit Answer", type="primary", use_container_width=True):
            if answer is None:
                st.warning("Please select an answer.")
                return
            correct_idx = question.get("correct", 1)
            is_correct = choices.index(answer) == correct_idx
            _process_answer(question, is_correct)

    elif q_type == "explain":
        st.text_area(
            "Your explanation:",
            key=f"explain_answer_{question['id']}",
            height=120,
        )
        rating = st.radio(
            "Rate your confidence (1 = no idea, 5 = nailed it):",
            [1, 2, 3, 4, 5],
            index=None,
            horizontal=True,
            key=f"explain_rating_{question['id']}",
        )
        if st.button("Submit Answer", type="primary", use_container_width=True):
            if rating is None:
                st.warning("Please rate your answer.")
                return
            is_correct = rating >= 3
            _process_answer(question, is_correct, self_rating=rating)


def _process_answer(
    question: dict, is_correct: bool, self_rating: int | None = None
) -> None:
    """Record the answer and transition to explanation view."""
    progress = _load_progress()
    record_answer(progress, question["id"], is_correct, self_rating=self_rating)
    _save(progress)

    st.session_state.quiz_answers.append(
        {
            "question_id": question["id"],
            "correct": is_correct,
            "topic": question.get("topic", "unknown"),
        }
    )
    st.session_state.show_explanation = True
    st.session_state.last_correct = is_correct
    st.rerun()


def _render_explanation(question: dict) -> None:
    """Render the explanation card after answering."""
    if st.session_state.last_correct:
        st.success("Correct!")
    else:
        st.error("Incorrect.")

    explanation = question.get("explanation", "No explanation available.")
    q_type = question.get("type", "multiple_choice")
    choices = question.get("choices", [])
    correct_idx = question.get("correct")

    if q_type in ("multiple_choice", "true_false") and choices and correct_idx is not None:
        st.markdown(f"**Correct answer:** {choices[correct_idx]}")

    st.markdown(f"**Explanation:** {explanation}")

    st.divider()

    total = len(st.session_state.quiz_questions)
    is_last = st.session_state.quiz_index >= total - 1

    if is_last:
        if st.button(
            "See Results", type="primary", use_container_width=True
        ):
            # Record the session
            progress = _load_progress()
            record_session(progress, st.session_state.quiz_answers)
            _save(progress)
            # Move past the last question to trigger summary
            st.session_state.quiz_index += 1
            st.session_state.show_explanation = False
            st.rerun()
    else:
        if st.button(
            "Next Question", type="primary", use_container_width=True
        ):
            st.session_state.quiz_index += 1
            st.session_state.show_explanation = False
            st.rerun()


def _render_summary() -> None:
    """Render the session summary after the last question."""
    st.subheader("Session Complete")
    st.divider()

    progress = _load_progress()
    summary = get_session_summary(progress, st.session_state.quiz_answers)

    total = summary["total"]
    correct = summary["correct"]
    accuracy = summary["accuracy"]
    pct = f"{accuracy:.0%}"

    # Score headline
    if accuracy >= 0.8:
        st.success(f"Score: {correct}/{total} correct ({pct})")
    elif accuracy >= 0.6:
        st.warning(f"Score: {correct}/{total} correct ({pct})")
    else:
        st.error(f"Score: {correct}/{total} correct ({pct})")

    # Per-topic breakdown
    per_topic = summary.get("per_topic", {})
    if per_topic:
        st.markdown("**Per-topic breakdown**")
        rows = []
        for topic, data in sorted(per_topic.items()):
            rows.append(
                {
                    "Topic": _display_topic(topic),
                    "Questions": data["total"],
                    "Correct": data["correct"],
                    "Accuracy": f"{data['accuracy']:.0%}",
                }
            )
        st.table(rows)

    st.divider()

    if st.button(
        "Start New Quiz", type="primary", use_container_width=True
    ):
        st.session_state.quiz_active = False
        st.session_state.quiz_questions = []
        st.session_state.quiz_index = 0
        st.session_state.quiz_answers = []
        st.session_state.show_explanation = False
        st.session_state.last_correct = False
        st.rerun()


def tab_quiz(questions: list[dict], progress: dict) -> None:
    """Main quiz tab logic."""
    if not st.session_state.quiz_active:
        _render_setup(questions, progress)
        return

    total = len(st.session_state.quiz_questions)
    idx = st.session_state.quiz_index

    # Past the last question -- show summary
    if idx >= total:
        _render_summary()
        return

    question = st.session_state.quiz_questions[idx]

    if st.session_state.show_explanation:
        _render_explanation(question)
    else:
        _render_question(question, idx, total)


# ---------------------------------------------------------------------------
# Tab 2: Progress
# ---------------------------------------------------------------------------
def tab_progress(questions: list[dict], progress: dict) -> None:
    """Render the progress dashboard."""
    total_questions = len(questions)
    cards = progress.get("cards", {})

    total_answered = sum(c["total_attempts"] for c in cards.values())
    total_correct = sum(c["total_correct"] for c in cards.values())
    overall_accuracy = total_correct / total_answered if total_answered > 0 else 0.0
    streak = compute_streak(progress)
    mastered_count = sum(
        1
        for c in cards.values()
        if c["streak"] >= 3 and c["ease_factor"] >= 2.5
    )

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Answered", total_answered)
    col2.metric("Overall Accuracy", f"{overall_accuracy:.0%}")
    col3.metric("Study Streak", f"{streak} day{'s' if streak != 1 else ''}")
    col4.metric("Mastered", f"{mastered_count}/{total_questions}")

    st.divider()

    # Per-topic table
    st.subheader("Topic Breakdown")

    topic_stats = get_topic_stats(progress, questions)
    all_topics = get_all_topics(questions)

    rows = []
    for topic in all_topics:
        td = topic_stats.get(topic, None)
        if td is None or td["attempts"] == 0:
            status = "-- (not started)"
            acc_display = "--"
            attempted = 0
            due = 0
        else:
            acc = td["accuracy"]
            acc_display = f"{acc:.0%}"
            attempted = td["attempts"]
            due = td["due"]
            if acc >= 0.75:
                status = "[+] Strong"
            elif acc >= 0.60:
                status = "[~] Developing"
            else:
                status = "[-] Needs Work"

        rows.append(
            {
                "Topic": _display_topic(topic),
                "Attempted": attempted,
                "Accuracy": acc_display,
                "Status": status,
                "Due for Review": due,
            }
        )

    if rows:
        st.table(rows)
    else:
        st.info("No progress data yet. Start a quiz to begin tracking.")

    due_total = get_due_count(progress)
    if due_total > 0:
        st.info(f"{due_total} question(s) due for review across all topics.")


# ---------------------------------------------------------------------------
# Tab 3: Browse
# ---------------------------------------------------------------------------
def tab_browse(questions: list[dict]) -> None:
    """Render the question browser with filters."""
    all_topics = get_all_topics(questions)
    topic_options = ["All"] + [_display_topic(t) for t in all_topics]

    col1, col2, col3 = st.columns(3)
    with col1:
        topic_filter = st.selectbox("Topic", topic_options, key="browse_topic")
    with col2:
        difficulties = sorted(set(q.get("difficulty", 1) for q in questions))
        diff_options = ["All"] + [str(d) for d in difficulties]
        diff_filter = st.selectbox("Difficulty", diff_options, key="browse_diff")
    with col3:
        types = sorted(set(q.get("type", "mc") for q in questions))
        type_options = ["All"] + types
        type_filter = st.selectbox("Type", type_options, key="browse_type")

    st.divider()

    # Apply filters
    filtered = list(questions)

    if topic_filter != "All":
        raw_topic = all_topics[topic_options.index(topic_filter) - 1]
        filtered = [q for q in filtered if q["topic"] == raw_topic]

    if diff_filter != "All":
        d = int(diff_filter)
        filtered = [q for q in filtered if q.get("difficulty", 1) == d]

    if type_filter != "All":
        filtered = [q for q in filtered if q.get("type", "mc") == type_filter]

    if not filtered:
        st.info("No questions match the selected filters.")
        return

    st.caption(f"Showing {len(filtered)} question(s)")

    for q in filtered:
        topic_display = _display_topic(q.get("topic", "general"))
        difficulty = q.get("difficulty", 1)
        label = f"{q['question']}"

        with st.expander(label):
            st.caption(
                f"Topic: {topic_display}  |  "
                f"Difficulty: {_difficulty_label(difficulty)}  |  "
                f"Type: {q.get('type', 'multiple_choice').replace('_', ' ').title()}"
            )

            q_type = q.get("type", "multiple_choice")
            choices = q.get("choices", [])
            correct_idx = q.get("correct")

            if q_type == "multiple_choice" and choices:
                for i, choice in enumerate(choices):
                    if i == correct_idx:
                        st.markdown(f"- **{choice}** (correct)")
                    else:
                        st.markdown(f"- {choice}")

            elif q_type == "true_false" and choices and correct_idx is not None:
                st.markdown(f"**Correct answer:** {choices[correct_idx]}")

            elif q_type == "explain":
                st.markdown("*(Open-ended -- self-rated)*")

            explanation = q.get("explanation", "")
            if explanation:
                st.markdown(f"**Explanation:** {explanation}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the Streamlit app."""
    st.title("Quant Finance Quiz")

    if not QUESTIONS_PATH.exists():
        st.error(
            f"Question bank not found at {QUESTIONS_PATH}. "
            "Please create questions.json in the quiz/ directory."
        )
        return

    questions = _load_questions()
    if not questions:
        st.error("Question bank is empty.")
        return

    progress = _load_progress()

    tab1, tab2, tab3 = st.tabs(["Quiz", "Progress", "Browse"])

    with tab1:
        tab_quiz(questions, progress)

    with tab2:
        tab_progress(questions, progress)

    with tab3:
        tab_browse(questions)


if __name__ == "__main__":
    main()
