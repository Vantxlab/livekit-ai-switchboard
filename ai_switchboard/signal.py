from __future__ import annotations


class Signal:
    """Named constants for heuristic routing signals."""

    # Complexity
    LONG_INPUT = "long_input"
    MULTI_QUESTION = "multi_question"
    COMPLEXITY_WORDS = "complexity_words"

    # Friction
    PUSHBACK = "pushback"
    FRUSTRATION = "frustration"
    REPEAT_REQUEST = "repeat_request"

    # Conversation
    INTERRUPTION = "interruption"

    # Topic (developer-defined)
    TOPIC_MATCH = "topic_match"


SIGNAL_WEIGHTS: dict[str, float] = {
    Signal.LONG_INPUT: 0.20,
    Signal.MULTI_QUESTION: 0.30,
    Signal.COMPLEXITY_WORDS: 0.20,
    Signal.PUSHBACK: 0.40,
    Signal.FRUSTRATION: 0.35,
    Signal.REPEAT_REQUEST: 0.25,
    Signal.INTERRUPTION: 0.20,
    Signal.TOPIC_MATCH: 0.50,
}

SIGNAL_WORDS: dict[str, list[str]] = {
    Signal.COMPLEXITY_WORDS: [
        "why",
        "explain",
        "compare",
        "difference",
        "what if",
        "walk me through",
        "how does",
    ],
    Signal.PUSHBACK: [
        "no",
        "that's wrong",
        "i already",
        "you said",
        "that's not",
        "incorrect",
        "not what i",
    ],
    Signal.FRUSTRATION: [
        "i don't understand",
        "doesn't make sense",
        "confused",
        "this isn't",
        "not helpful",
    ],
    Signal.REPEAT_REQUEST: [
        "say that again",
        "can you repeat",
        "didn't catch",
        "one more time",
        "come again",
        "pardon",
    ],
}
