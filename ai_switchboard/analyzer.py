from __future__ import annotations

from .config import SwitchboardConfig
from .context import Context
from .signal import SIGNAL_WEIGHTS, SIGNAL_WORDS, Signal


class HeuristicAnalyzer:
    """Scores a conversation turn by detecting heuristic signals."""

    def analyze(self, ctx: Context, config: SwitchboardConfig) -> Context:
        """Detect signals and compute a weighted heuristic score.

        Mutates ``ctx.signals_fired`` and ``ctx.heuristic_score``, then
        returns the same context object for convenience.
        """
        signals: list[str] = []
        message_lower = ctx.last_message.lower()

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

        # --- Topic signals ---
        if config.smart_topics and any(
            topic.lower() in message_lower for topic in config.smart_topics
        ):
            signals.append(Signal.TOPIC_MATCH)

        # Compute score
        score = sum(SIGNAL_WEIGHTS.get(s, 0.0) for s in signals)
        ctx.signals_fired = signals
        ctx.heuristic_score = min(score, 1.0)
        return ctx

    @staticmethod
    def _match_words(message_lower: str, signal_key: str) -> bool:
        """Return True if any phrase for *signal_key* appears in the message."""
        phrases = SIGNAL_WORDS.get(signal_key, [])
        return any(phrase in message_lower for phrase in phrases)
