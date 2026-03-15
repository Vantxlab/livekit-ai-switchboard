from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SwitchEvent:
    """Emitted on every routing decision for observability."""

    turn: int
    from_model: str
    to_model: str
    triggered_by: str
    signals_fired: list[str] = field(default_factory=list)
    heuristic_score: float = 0.0
    changed: bool = False
