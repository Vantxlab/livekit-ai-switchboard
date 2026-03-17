"""Tests for #5 — Built-in SwitchboardMetrics collector."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

from ai_switchboard.config import SwitchboardConfig
from ai_switchboard.events import SwitchEvent
from ai_switchboard.metrics import SwitchboardMetrics
from ai_switchboard.switchboard import Switchboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(name: str = "mock-model") -> MagicMock:
    mock = MagicMock()
    type(mock).model = PropertyMock(return_value=name)
    mock.chat.return_value = MagicMock(name="LLMStream")
    return mock


def _make_models(names: list[str] | None = None) -> dict[str, MagicMock]:
    names = names or ["fast", "standard", "premium"]
    return {n: _make_mock_llm(f"{n}-model") for n in names}


def _make_chat_ctx(messages: list[tuple[str, str]] | None = None) -> MagicMock:
    chat_ctx = MagicMock()
    msgs = []
    for role, text in messages or [("user", "hello")]:
        msg = MagicMock()
        msg.role = role
        msg.text_content = text
        msg.transcript_confidence = None
        msgs.append(msg)
    chat_ctx.messages = msgs
    return chat_ctx


def _chat(sb: Switchboard, text: str = "hello"):
    ctx = _make_chat_ctx([("user", text)])
    return sb.chat(chat_ctx=ctx)


# ---------------------------------------------------------------------------
# Unit tests for SwitchboardMetrics dataclass
# ---------------------------------------------------------------------------


class TestSwitchboardMetricsUnit:
    def test_initial_state(self):
        m = SwitchboardMetrics()
        assert m.total_turns == 0
        assert m.turns_per_model == {}
        assert m.switches_per_trigger == {}
        assert m.total_switches == 0
        assert m.avg_heuristic_score == 0.0

    def test_record_turn_no_change(self):
        m = SwitchboardMetrics()
        m.record_turn(model="fast", triggered_by="default", heuristic_score=0.1, changed=False)
        assert m.total_turns == 1
        assert m.turns_per_model == {"fast": 1}
        assert m.total_switches == 0
        assert m.switches_per_trigger == {}
        assert m.avg_heuristic_score == 0.1

    def test_record_turn_with_change(self):
        m = SwitchboardMetrics()
        m.record_turn(model="fast", triggered_by="default", heuristic_score=0.0, changed=False)
        m.record_turn(model="premium", triggered_by="topic", heuristic_score=0.5, changed=True)
        assert m.total_turns == 2
        assert m.turns_per_model == {"fast": 1, "premium": 1}
        assert m.total_switches == 1
        assert m.switches_per_trigger == {"topic": 1}
        assert m.avg_heuristic_score == 0.25

    def test_reset(self):
        m = SwitchboardMetrics()
        m.record_turn(model="fast", triggered_by="default", heuristic_score=0.5, changed=False)
        m.reset()
        assert m.total_turns == 0
        assert m.turns_per_model == {}
        assert m.avg_heuristic_score == 0.0


# ---------------------------------------------------------------------------
# Integration tests — metrics via Switchboard
# ---------------------------------------------------------------------------


class TestMetricsIntegration:
    def test_metrics_property_exists(self):
        models = _make_models()
        sb = Switchboard(models=models)
        assert isinstance(sb.metrics, SwitchboardMetrics)

    def test_turns_tracked(self):
        models = _make_models()
        sb = Switchboard(models=models)
        _chat(sb, "Hello")
        _chat(sb, "Hi again")
        assert sb.metrics.total_turns == 2
        assert sb.metrics.turns_per_model["fast"] == 2

    def test_switch_tracked(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
            ),
        )
        _chat(sb, "Hello")
        _chat(sb, "Tell me about pricing")
        assert sb.metrics.total_turns == 2
        assert sb.metrics.total_switches == 1
        assert sb.metrics.turns_per_model["fast"] == 1
        assert sb.metrics.turns_per_model["premium"] == 1
        assert "topic" in sb.metrics.switches_per_trigger

    def test_avg_score_tracked(self):
        models = _make_models()
        sb = Switchboard(models=models)
        _chat(sb, "Hello")  # score ~0.0
        # pushback (0.40) + multi_question (0.30) = 0.70
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb.metrics.avg_heuristic_score > 0.0

    def test_reset_clears_metrics(self):
        models = _make_models()
        sb = Switchboard(models=models)
        _chat(sb, "Hello")
        assert sb.metrics.total_turns == 1
        sb.reset()
        assert sb.metrics.total_turns == 0
        assert sb.metrics.turns_per_model == {}

    def test_multiple_triggers_tracked(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
                escalation_model="premium",
                escalation_threshold=0.6,
                cooldown_turns=0,
            ),
        )
        # topic switch
        _chat(sb, "Tell me about pricing")
        assert sb.metrics.total_switches == 1

        # back to fast (cooldown=0)
        _chat(sb, "ok thanks")

        # heuristic switch: pushback (0.40) + multi_question (0.30)
        _chat(sb, "No, that's wrong. Why? How?")

        assert sb.metrics.total_switches >= 2
        # Should have both topic and heuristic triggers
        assert len(sb.metrics.switches_per_trigger) >= 1
