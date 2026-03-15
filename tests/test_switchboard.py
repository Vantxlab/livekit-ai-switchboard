from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, PropertyMock

from ai_switchboard.config import SwitchboardConfig
from ai_switchboard.events import SwitchEvent
from ai_switchboard.rule import Rule
from ai_switchboard.switchboard import Switchboard


# ---------------------------------------------------------------------------
# Helpers — lightweight mocks that quack like livekit llm.LLM
# ---------------------------------------------------------------------------


def _make_mock_llm(name: str = "mock-model") -> MagicMock:
    """Return a MagicMock that behaves enough like ``llm.LLM`` for routing tests."""
    mock = MagicMock()
    type(mock).model = PropertyMock(return_value=name)
    mock.chat.return_value = MagicMock(name="LLMStream")
    return mock


def _make_chat_ctx(messages: list[tuple[str, str]] | None = None) -> MagicMock:
    """Build a fake ChatContext with a list of (role, text) tuples."""
    chat_ctx = MagicMock()
    msgs = []
    for role, text in messages or [("user", "hello")]:
        msg = MagicMock()
        msg.role = role
        msg.text_content = text
        msgs.append(msg)
    chat_ctx.messages = msgs
    return chat_ctx


def _chat(sb: Switchboard, text: str = "hello") -> Any:
    """Convenience wrapper to call ``sb.chat()`` with a simple user message."""
    ctx = _make_chat_ctx([("user", text)])
    return sb.chat(chat_ctx=ctx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicRouting:
    def test_starts_on_fast_by_default(self):
        fast, smart = _make_mock_llm("fast-model"), _make_mock_llm("smart-model")
        sb = Switchboard(fast=fast, smart=smart)
        assert sb.current_model == "fast"

    def test_simple_message_stays_fast(self):
        fast, smart = _make_mock_llm("fast-model"), _make_mock_llm("smart-model")
        sb = Switchboard(fast=fast, smart=smart)
        _chat(sb, "Hi there!")
        assert sb.current_model == "fast"
        fast.chat.assert_called_once()
        smart.chat.assert_not_called()

    def test_complex_message_escalates_to_smart(self):
        fast, smart = _make_mock_llm("fast-model"), _make_mock_llm("smart-model")
        sb = Switchboard(fast=fast, smart=smart)
        # pushback (0.40) + multi_question (0.30) = 0.70 > 0.60 threshold
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb.current_model == "smart"
        smart.chat.assert_called_once()

    def test_provider_is_switchboard(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        assert sb.provider == "switchboard"


class TestStartOn:
    def test_start_on_smart(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast,
            smart=smart,
            config=SwitchboardConfig(start_on="smart"),
        )
        assert sb.current_model == "smart"
        _chat(sb, "Hello")
        smart.chat.assert_called_once()


class TestDeescalation:
    def test_deescalates_after_cooldown(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast,
            smart=smart,
            config=SwitchboardConfig(cooldown_turns=2),
        )
        # Escalate
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb.current_model == "smart"
        # Two more turns on smart (cooldown = 2)
        _chat(sb, "ok")
        _chat(sb, "yes")
        # Now should de-escalate
        _chat(sb, "thanks")
        assert sb.current_model == "fast"

    def test_cooldown_prevents_early_deescalation(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast,
            smart=smart,
            config=SwitchboardConfig(cooldown_turns=3),
        )
        # Escalate
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb.current_model == "smart"
        # Only 1 turn — still in cooldown
        _chat(sb, "ok")
        assert sb.current_model == "smart"


class TestCustomRules:
    def test_rule_overrides_heuristic(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        # Rule forces smart for any message containing "vip"
        rule = Rule(
            name="vip",
            condition=lambda ctx: "vip" in ctx.last_message.lower(),
            use="smart",
            priority=10,
        )
        sb = Switchboard(fast=fast, smart=smart, rules=[rule])
        _chat(sb, "I am a VIP customer")
        assert sb.current_model == "smart"

    def test_rule_forces_fast(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        # Force fast for short confirmations even when on smart
        rule = Rule(
            name="short_confirm",
            condition=lambda ctx: ctx.last_message_word_count < 3,
            use="fast",
        )
        sb = Switchboard(
            fast=fast,
            smart=smart,
            config=SwitchboardConfig(start_on="smart", cooldown_turns=0),
            rules=[rule],
        )
        _chat(sb, "yes")
        assert sb.current_model == "fast"

    def test_higher_priority_rule_wins(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        rules = [
            Rule(
                name="always_smart", condition=lambda ctx: True, use="smart", priority=1
            ),
            Rule(
                name="always_fast", condition=lambda ctx: True, use="fast", priority=10
            ),
        ]
        sb = Switchboard(fast=fast, smart=smart, rules=rules)
        _chat(sb, "hello")
        assert sb.current_model == "fast"


class TestTopicEscalation:
    def test_topic_match_escalates(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast,
            smart=smart,
            config=SwitchboardConfig(
                smart_topics=["pricing", "warranty"],
                escalation_threshold=0.5,
            ),
        )
        _chat(sb, "What is the pricing for this?")
        assert sb.current_model == "smart"


class TestSwitchEventCallback:
    def test_on_switch_called(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast,
            smart=smart,
            config=SwitchboardConfig(on_switch=events.append),
        )
        _chat(sb, "Hi")
        assert len(events) == 1
        assert events[0].changed is False
        assert events[0].to_model == "fast"

    def test_on_switch_records_change(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast,
            smart=smart,
            config=SwitchboardConfig(on_switch=events.append),
        )
        _chat(sb, "No, that's wrong. Why? How?")
        assert len(events) == 1
        assert events[0].changed is True
        assert events[0].from_model == "fast"
        assert events[0].to_model == "smart"


class TestInterruption:
    def test_record_interruption(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        sb.record_interruption()
        # interruption (0.20) alone won't escalate (< 0.60)
        _chat(sb, "hello")
        assert sb.current_model == "fast"

    def test_interruption_resets_after_turn(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        sb.record_interruption()
        _chat(sb, "hello")
        # After the turn, interruption_count should be reset
        # So next turn should not see interruption signal
        events: list[SwitchEvent] = []
        sb._config.on_switch = events.append
        _chat(sb, "hi again")
        assert "interruption" not in events[0].signals_fired


class TestModelProperty:
    def test_model_reflects_current(self):
        fast, smart = _make_mock_llm("fast-v1"), _make_mock_llm("smart-v1")
        sb = Switchboard(fast=fast, smart=smart)
        assert sb.model == "fast-v1"
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb.model == "smart-v1"
