from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .events import SwitchEvent


@dataclass
class SwitchboardConfig:
    """Tuning knobs for Switchboard routing behavior."""

    default_model: str = ""  # fallback model; "" = first in models dict
    cooldown_turns: int = 2

    # Topic routing: model_name -> keyword list
    model_topics: dict[str, list[str]] = field(default_factory=dict)

    # Heuristic auto-escalation
    escalation_model: str = ""  # model to escalate to; "" = disabled
    escalation_threshold: float = 0.6
    min_signals_for_escalation: int = 1  # require N+ signals to escalate

    # Signal weight overrides (merged on top of SIGNAL_WEIGHTS defaults)
    signal_weights: dict[str, float] | None = None

    # Voice thresholds
    stt_confidence_threshold: float = 0.7
    long_audio_threshold: float = 10.0  # seconds

    # Context window
    context_window_size: int = 5  # number of recent user messages to keep

    # Latency-aware routing
    max_ttfb_ms: dict[str, float] | None = None  # model_name -> max avg TTFB
    timeout_fallback_model: str = ""  # model to fall back to when TTFB exceeded
    ttfb_window_size: int = 10  # rolling window length for TTFB tracking

    # Observability
    on_switch: Optional[Callable[[SwitchEvent], None]] = (
        None  # fires only on model change
    )
    on_decision: Optional[Callable[[SwitchEvent], None]] = None  # fires every turn
    log_decisions: bool = False
