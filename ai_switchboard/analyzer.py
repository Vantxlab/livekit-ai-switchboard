from __future__ import annotations

import re

from .config import SwitchboardConfig
from .context import Context
from .signal import SIGNAL_WEIGHTS, SIGNAL_WORDS, Signal


class HeuristicAnalyzer:
    """Scores a conversation turn by detecting heuristic signals."""

    def __init__(self) -> None:
        # Pre-compile one regex per signal key for word-boundary matching.
        self._compiled: dict[str, re.Pattern[str]] = {}
        for signal_key, phrases in SIGNAL_WORDS.items():
            parts: list[str] = []
            for phrase in phrases:
                words = phrase.split()
                if len(words) == 1:
                    parts.append(r"\b" + re.escape(phrase) + r"\b")
                else:
                    # Multi-word: join with flexible whitespace, word boundaries
                    escaped = r"\s+".join(re.escape(w) for w in words)
                    parts.append(r"\b" + escaped + r"\b")
            self._compiled[signal_key] = re.compile("|".join(parts), re.IGNORECASE)

    def analyze(self, ctx: Context, config: SwitchboardConfig) -> Context:
        """Detect signals and compute a weighted heuristic score.

        Mutates ``ctx.signals_fired`` and ``ctx.heuristic_score``, then
        returns the same context object for convenience.
        """
        signals: list[str] = []
        message_lower = ctx.last_message.lower()

        # Resolve effective weights
        weights = SIGNAL_WEIGHTS
        if config.signal_weights:
            weights = {**SIGNAL_WEIGHTS, **config.signal_weights}

        # --- Complexity signals ---
        if ctx.last_message_word_count > 25:
            signals.append(Signal.LONG_INPUT)

        if ctx.last_message.count("?") >= 2:
            signals.append(Signal.MULTI_QUESTION)

        if self._match_words(message_lower, Signal.COMPLEXITY_WORDS):
            signals.append(Signal.COMPLEXITY_WORDS)

        # --- Friction signals ---
        if self._match_words(message_lower, Signal.PUSHBACK):
            signals.append(Signal.PUSHBACK)

        if self._match_words(message_lower, Signal.FRUSTRATION):
            signals.append(Signal.FRUSTRATION)

        if self._match_words(message_lower, Signal.REPEAT_REQUEST):
            signals.append(Signal.REPEAT_REQUEST)

        # --- Conversation signals ---
        if ctx.interruption_count > 0:
            signals.append(Signal.INTERRUPTION)

        # --- Voice signals ---
        if (
            ctx.stt_confidence is not None
            and ctx.stt_confidence < config.stt_confidence_threshold
        ):
            signals.append(Signal.LOW_STT_CONFIDENCE)

        if (
            ctx.audio_duration is not None
            and ctx.audio_duration > config.long_audio_threshold
        ):
            signals.append(Signal.LONG_AUDIO_TURN)

        # --- Topic signals ---
        all_topics: list[str] = []
        for topics in config.model_topics.values():
            all_topics.extend(topics)
        if all_topics and self._match_topic(message_lower, all_topics):
            signals.append(Signal.TOPIC_MATCH)

        # Compute score
        score = sum(weights.get(s, 0.0) for s in signals)
        ctx.signals_fired = signals
        ctx.heuristic_score = min(score, 1.0)
        return ctx

    def _match_words(self, message_lower: str, signal_key: str) -> bool:
        """Return True if any phrase for *signal_key* appears in the message."""
        pattern = self._compiled.get(signal_key)
        if pattern is None:
            return False
        return pattern.search(message_lower) is not None

    @staticmethod
    def _match_topic(message_lower: str, topics: list[str]) -> bool:
        """Return True if any topic matches with word boundaries."""
        for topic in topics:
            words = topic.lower().split()
            if len(words) == 1:
                if re.search(r"\b" + re.escape(topic.lower()) + r"\b", message_lower):
                    return True
            else:
                escaped = r"\s+".join(re.escape(w) for w in words)
                if re.search(r"\b" + escaped + r"\b", message_lower):
                    return True
        return False
