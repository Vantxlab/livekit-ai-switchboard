from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SwitchboardMetrics:
    """Aggregated routing statistics."""

    total_turns: int = 0
    turns_per_model: dict[str, int] = field(default_factory=dict)
    switches_per_trigger: dict[str, int] = field(default_factory=dict)
    total_switches: int = 0
    _score_sum: float = 0.0

    @property
    def avg_heuristic_score(self) -> float:
        """Average heuristic score across all turns."""
        if self.total_turns == 0:
            return 0.0
        return self._score_sum / self.total_turns

    def record_turn(
        self,
        model: str,
        triggered_by: str,
        heuristic_score: float,
        changed: bool,
    ) -> None:
        """Record a single routing decision."""
        self.total_turns += 1
        self.turns_per_model[model] = self.turns_per_model.get(model, 0) + 1
        self._score_sum += heuristic_score
        if changed:
            self.total_switches += 1
            self.switches_per_trigger[triggered_by] = (
                self.switches_per_trigger.get(triggered_by, 0) + 1
            )

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_turns = 0
        self.turns_per_model.clear()
        self.switches_per_trigger.clear()
        self.total_switches = 0
        self._score_sum = 0.0
