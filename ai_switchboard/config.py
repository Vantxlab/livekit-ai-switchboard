from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .events import SwitchEvent


@dataclass
class SwitchboardConfig:
    """Tuning knobs for Switchboard routing behavior."""

    default_model: str = ""                    # fallback model; "" = first in models dict
    cooldown_turns: int = 2

    # Topic routing: model_name -> keyword list
    model_topics: dict[str, list[str]] = field(default_factory=dict)

    # Heuristic auto-escalation
    escalation_model: str = ""                 # model to escalate to; "" = disabled
    escalation_threshold: float = 0.6

    # Voice thresholds
    stt_confidence_threshold: float = 0.7
    long_audio_threshold: float = 10.0         # seconds

    # Observability
    on_switch: Optional[Callable[[SwitchEvent], None]] = None       # fires only on model change
    on_decision: Optional[Callable[[SwitchEvent], None]] = None     # fires every turn
    log_decisions: bool = False
