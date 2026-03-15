from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .events import SwitchEvent


@dataclass
class SwitchboardConfig:
    """Tuning knobs for Switchboard routing behavior."""

    escalation_threshold: float = 0.60
    deescalation_threshold: float = 0.20
    cooldown_turns: int = 2
    start_on: Literal["fast", "smart"] = "fast"
    smart_topics: list[str] = field(default_factory=list)

    # Observability
    on_switch: Optional[Callable[[SwitchEvent], None]] = None
    log_decisions: bool = False
