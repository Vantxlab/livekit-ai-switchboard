from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Context:
    """Snapshot of conversation state passed to every rule."""

    # Current turn
    last_message: str = ""
    last_message_word_count: int = 0
    turn_count: int = 0

    # Conversation health
    interruption_count: int = 0
    repeat_request_count: int = 0

    # Routing state
    current_model: str = "fast"
    turns_on_current_model: int = 0
    last_switch_turn: int = -1

    # Voice
    stt_confidence: float | None = None
    audio_duration: float | None = None

    # Populated by analyzer
    signals_fired: list[str] = field(default_factory=list)
    heuristic_score: float = 0.0
