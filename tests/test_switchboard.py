from __future__ import annotations

import logging
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


def _make_chat_ctx_with_items(messages: list[tuple[str, str]] | None = None) -> MagicMock:
    """Build a fake ChatContext using .items (like real livekit ChatContext)."""
    chat_ctx = MagicMock()
    msgs = []
    for role, text in messages or [("user", "hello")]:
        msg = MagicMock()
        msg.role = role
        msg.text_content = text
        msgs.append(msg)
    chat_ctx.items = msgs
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


class TestChatContextItems:
    """Test the .items code path (real livekit ChatContext uses .items)."""

    def test_items_path_routes_correctly(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        ctx = _make_chat_ctx_with_items([("user", "Hi there!")])
        sb.chat(chat_ctx=ctx)
        assert sb.current_model == "fast"
        fast.chat.assert_called_once()

    def test_items_path_escalates(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        ctx = _make_chat_ctx_with_items([("user", "No, that's wrong. Why? How?")])
        sb.chat(chat_ctx=ctx)
        assert sb.current_model == "smart"

    def test_no_user_message_stays_on_current(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        ctx = _make_chat_ctx([("assistant", "Hello! How can I help?")])
        sb.chat(chat_ctx=ctx)
        assert sb.current_model == "fast"

    def test_picks_last_user_message(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        ctx = _make_chat_ctx([
            ("user", "Hi"),
            ("assistant", "Hello!"),
            ("user", "No, that's wrong. Why? How?"),
        ])
        sb.chat(chat_ctx=ctx)
        assert sb.current_model == "smart"


class TestRepeatRequestAccumulation:
    def test_repeat_request_count_increments(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        assert sb._repeat_request_count == 0
        _chat(sb, "Can you say that again please?")
        assert sb._repeat_request_count == 1
        _chat(sb, "Can you repeat that?")
        assert sb._repeat_request_count == 2

    def test_non_repeat_does_not_increment(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        _chat(sb, "Hello there")
        assert sb._repeat_request_count == 0


class TestTurnTracking:
    def test_turn_count_increments(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        assert sb._turn_count == 0
        _chat(sb, "Hello")
        assert sb._turn_count == 1
        _chat(sb, "Hi again")
        assert sb._turn_count == 2

    def test_turns_on_current_increments_without_switch(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        _chat(sb, "Hello")
        assert sb._turns_on_current == 1
        _chat(sb, "Hi again")
        assert sb._turns_on_current == 2

    def test_turns_on_current_resets_on_switch(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        _chat(sb, "Hello")
        assert sb._turns_on_current == 1
        # Escalate
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb._turns_on_current == 0
        assert sb._last_switch_turn == 1


class TestSwitchEventDetails:
    def test_event_contains_heuristic_score(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(on_switch=events.append),
        )
        _chat(sb, "No, that's wrong. Why? How?")
        assert events[0].heuristic_score >= 0.60

    def test_event_triggered_by_heuristic(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(on_switch=events.append),
        )
        _chat(sb, "No, that's wrong. Why? How?")
        assert events[0].triggered_by == "heuristic"

    def test_event_triggered_by_rule_name(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_mock_llm(), _make_mock_llm()
        rule = Rule(
            name="billing",
            condition=lambda ctx: "billing" in ctx.last_message.lower(),
            use="smart",
        )
        sb = Switchboard(
            fast=fast, smart=smart,
            rules=[rule],
            config=SwitchboardConfig(on_switch=events.append),
        )
        _chat(sb, "I have a billing question")
        assert events[0].triggered_by == "billing"

    def test_event_turn_number(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(on_switch=events.append),
        )
        _chat(sb, "Hello")
        _chat(sb, "Hi again")
        assert events[0].turn == 0
        assert events[1].turn == 1

    def test_event_signals_list(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(on_switch=events.append),
        )
        _chat(sb, "No, that's wrong. Why? How?")
        assert "pushback" in events[0].signals_fired
        assert "multi_question" in events[0].signals_fired


class TestLogDecisions:
    def test_log_decisions_emits_log(self, caplog):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(log_decisions=True),
        )
        with caplog.at_level(logging.INFO, logger="ai_switchboard"):
            _chat(sb, "Hello")
        assert len(caplog.records) == 1
        assert "turn=0" in caplog.records[0].message
        assert "model=fast" in caplog.records[0].message

    def test_log_decisions_off_no_log(self, caplog):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(log_decisions=False),
        )
        with caplog.at_level(logging.INFO, logger="ai_switchboard"):
            _chat(sb, "Hello")
        assert len(caplog.records) == 0


class TestDeescalationThreshold:
    def test_score_between_thresholds_holds_current(self):
        """Score between deescalation and escalation thresholds keeps current model."""
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(start_on="smart", cooldown_turns=0),
        )
        # "pardon" triggers repeat_request (0.25), which is between
        # deescalation_threshold (0.20) and escalation_threshold (0.60)
        _chat(sb, "I beg your pardon?")
        assert sb.current_model == "smart"

    def test_very_low_score_deescalates(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(start_on="smart", cooldown_turns=0),
        )
        # "ok" triggers no signals → score 0.0 ≤ 0.20 → deescalate
        _chat(sb, "ok")
        assert sb.current_model == "fast"


class TestDefaultConfig:
    def test_default_config_applied(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        assert sb._config.escalation_threshold == 0.60
        assert sb._config.deescalation_threshold == 0.20
        assert sb._config.cooldown_turns == 2
        assert sb._config.start_on == "fast"
        assert sb._config.smart_topics == []
        assert sb._config.on_switch is None
        assert sb._config.log_decisions is False


class TestChatForwardsAllArgs:
    def test_tools_and_options_forwarded(self):
        fast, smart = _make_mock_llm(), _make_mock_llm()
        sb = Switchboard(fast=fast, smart=smart)
        ctx = _make_chat_ctx([("user", "hi")])
        mock_tools = [MagicMock()]
        sb.chat(chat_ctx=ctx, tools=mock_tools)
        _, kwargs = fast.chat.call_args
        assert kwargs["tools"] is mock_tools
        assert kwargs["chat_ctx"] is ctx
