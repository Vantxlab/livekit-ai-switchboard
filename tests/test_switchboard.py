from __future__ import annotations

import logging
from typing import Any

from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChatContext, ChoiceDelta
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from ai_switchboard.config import SwitchboardConfig
from ai_switchboard.events import SwitchEvent
from ai_switchboard.rule import Rule
from ai_switchboard.switchboard import Switchboard


# ---------------------------------------------------------------------------
# Helpers — fake LLM that implements the real livekit llm.LLM interface
# ---------------------------------------------------------------------------


class _FakeLLMStream(llm.LLMStream):
    """Minimal LLMStream that emits a single assistant chunk."""

    async def _run(self) -> None:
        self._event_ch.send_nowait(
            ChatChunk(
                id="fake-chunk",
                delta=ChoiceDelta(role="assistant", content="Hello!"),
            )
        )


class FakeLLM(llm.LLM):
    """A real ``llm.LLM`` subclass for testing — no network calls."""

    def __init__(self, *, model_name: str = "fake-model") -> None:
        super().__init__()
        self._model_name = model_name
        self.chat_call_count = 0

    @property
    def model(self) -> str:  # type: ignore[override]
        return self._model_name

    @property
    def provider(self) -> str:  # type: ignore[override]
        return "fake"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls=None,
        tool_choice=None,
        extra_kwargs=None,
    ) -> llm.LLMStream:
        self.chat_call_count += 1
        return _FakeLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


def _make_llm(name: str = "fake-model") -> FakeLLM:
    """Return a FakeLLM instance."""
    return FakeLLM(model_name=name)


def _make_chat_ctx(messages: list[tuple[str, str]] | None = None) -> ChatContext:
    """Build a real ChatContext with a list of (role, text) tuples."""
    ctx = ChatContext()
    for role, text in messages or [("user", "hello")]:
        ctx.add_message(role=role, content=text)
    return ctx


def _chat(sb: Switchboard, text: str = "hello") -> Any:
    """Convenience wrapper to call ``sb.chat()`` with a simple user message."""
    ctx = _make_chat_ctx([("user", text)])
    return sb.chat(chat_ctx=ctx)


# ---------------------------------------------------------------------------
# Tests — all async because LLMStream.__init__ requires a running event loop
# ---------------------------------------------------------------------------


class TestBasicRouting:
    async def test_starts_on_fast_by_default(self):
        fast, smart = _make_llm("fast-model"), _make_llm("smart-model")
        sb = Switchboard(fast=fast, smart=smart)
        assert sb.current_model == "fast"

    async def test_simple_message_stays_fast(self):
        fast, smart = _make_llm("fast-model"), _make_llm("smart-model")
        sb = Switchboard(fast=fast, smart=smart)
        _chat(sb, "Hi there!")
        assert sb.current_model == "fast"
        assert fast.chat_call_count == 1
        assert smart.chat_call_count == 0

    async def test_complex_message_escalates_to_smart(self):
        fast, smart = _make_llm("fast-model"), _make_llm("smart-model")
        sb = Switchboard(fast=fast, smart=smart)
        # pushback (0.40) + multi_question (0.30) = 0.70 > 0.60 threshold
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb.current_model == "smart"
        assert smart.chat_call_count == 1

    async def test_provider_is_switchboard(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        assert sb.provider == "switchboard"


class TestStartOn:
    async def test_start_on_smart(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(
            fast=fast,
            smart=smart,
            config=SwitchboardConfig(start_on="smart"),
        )
        assert sb.current_model == "smart"
        _chat(sb, "Hello")
        assert smart.chat_call_count == 1


class TestDeescalation:
    async def test_deescalates_after_cooldown(self):
        fast, smart = _make_llm(), _make_llm()
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

    async def test_cooldown_prevents_early_deescalation(self):
        fast, smart = _make_llm(), _make_llm()
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
    async def test_rule_overrides_heuristic(self):
        fast, smart = _make_llm(), _make_llm()
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

    async def test_rule_forces_fast(self):
        fast, smart = _make_llm(), _make_llm()
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

    async def test_higher_priority_rule_wins(self):
        fast, smart = _make_llm(), _make_llm()
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
    async def test_topic_match_escalates(self):
        fast, smart = _make_llm(), _make_llm()
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
    async def test_on_switch_called(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(
            fast=fast,
            smart=smart,
            config=SwitchboardConfig(on_switch=events.append),
        )
        _chat(sb, "Hi")
        assert len(events) == 1
        assert events[0].changed is False
        assert events[0].to_model == "fast"

    async def test_on_switch_records_change(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_llm(), _make_llm()
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
    async def test_record_interruption(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        sb.record_interruption()
        # interruption (0.20) alone won't escalate (< 0.60)
        _chat(sb, "hello")
        assert sb.current_model == "fast"

    async def test_interruption_resets_after_turn(self):
        fast, smart = _make_llm(), _make_llm()
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
    async def test_model_reflects_current(self):
        fast, smart = _make_llm("fast-v1"), _make_llm("smart-v1")
        sb = Switchboard(fast=fast, smart=smart)
        assert sb.model == "fast-v1"
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb.model == "smart-v1"


class TestChatContextVariants:
    """Test different ChatContext structures."""

    async def test_no_user_message_stays_on_current(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        ctx = _make_chat_ctx([("assistant", "Hello! How can I help?")])
        sb.chat(chat_ctx=ctx)
        assert sb.current_model == "fast"

    async def test_picks_last_user_message(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        ctx = _make_chat_ctx([
            ("user", "Hi"),
            ("assistant", "Hello!"),
            ("user", "No, that's wrong. Why? How?"),
        ])
        sb.chat(chat_ctx=ctx)
        assert sb.current_model == "smart"

    async def test_empty_chat_context(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        ctx = ChatContext()
        sb.chat(chat_ctx=ctx)
        assert sb.current_model == "fast"


class TestRepeatRequestAccumulation:
    async def test_repeat_request_count_increments(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        assert sb._repeat_request_count == 0
        _chat(sb, "Can you say that again please?")
        assert sb._repeat_request_count == 1
        _chat(sb, "Can you repeat that?")
        assert sb._repeat_request_count == 2

    async def test_non_repeat_does_not_increment(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        _chat(sb, "Hello there")
        assert sb._repeat_request_count == 0


class TestTurnTracking:
    async def test_turn_count_increments(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        assert sb._turn_count == 0
        _chat(sb, "Hello")
        assert sb._turn_count == 1
        _chat(sb, "Hi again")
        assert sb._turn_count == 2

    async def test_turns_on_current_increments_without_switch(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        _chat(sb, "Hello")
        assert sb._turns_on_current == 1
        _chat(sb, "Hi again")
        assert sb._turns_on_current == 2

    async def test_turns_on_current_resets_on_switch(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        _chat(sb, "Hello")
        assert sb._turns_on_current == 1
        # Escalate
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb._turns_on_current == 0
        assert sb._last_switch_turn == 1


class TestSwitchEventDetails:
    async def test_event_contains_heuristic_score(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(on_switch=events.append),
        )
        _chat(sb, "No, that's wrong. Why? How?")
        assert events[0].heuristic_score >= 0.60

    async def test_event_triggered_by_heuristic(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(on_switch=events.append),
        )
        _chat(sb, "No, that's wrong. Why? How?")
        assert events[0].triggered_by == "heuristic"

    async def test_event_triggered_by_rule_name(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_llm(), _make_llm()
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

    async def test_event_turn_number(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(on_switch=events.append),
        )
        _chat(sb, "Hello")
        _chat(sb, "Hi again")
        assert events[0].turn == 0
        assert events[1].turn == 1

    async def test_event_signals_list(self):
        events: list[SwitchEvent] = []
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(on_switch=events.append),
        )
        _chat(sb, "No, that's wrong. Why? How?")
        assert "pushback" in events[0].signals_fired
        assert "multi_question" in events[0].signals_fired


class TestLogDecisions:
    async def test_log_decisions_emits_log(self, caplog):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(log_decisions=True),
        )
        with caplog.at_level(logging.INFO, logger="ai_switchboard"):
            _chat(sb, "Hello")
        assert len(caplog.records) == 1
        assert "turn=0" in caplog.records[0].message
        assert "model=fast" in caplog.records[0].message

    async def test_log_decisions_off_no_log(self, caplog):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(log_decisions=False),
        )
        with caplog.at_level(logging.INFO, logger="ai_switchboard"):
            _chat(sb, "Hello")
        assert len(caplog.records) == 0


class TestDeescalationThreshold:
    async def test_score_between_thresholds_holds_current(self):
        """Score between deescalation and escalation thresholds keeps current model."""
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(start_on="smart", cooldown_turns=0),
        )
        # "pardon" triggers repeat_request (0.25), which is between
        # deescalation_threshold (0.20) and escalation_threshold (0.60)
        _chat(sb, "I beg your pardon?")
        assert sb.current_model == "smart"

    async def test_very_low_score_deescalates(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(
            fast=fast, smart=smart,
            config=SwitchboardConfig(start_on="smart", cooldown_turns=0),
        )
        # "ok" triggers no signals → score 0.0 ≤ 0.20 → deescalate
        _chat(sb, "ok")
        assert sb.current_model == "fast"


class TestDefaultConfig:
    async def test_default_config_applied(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        assert sb._config.escalation_threshold == 0.60
        assert sb._config.deescalation_threshold == 0.20
        assert sb._config.cooldown_turns == 2
        assert sb._config.start_on == "fast"
        assert sb._config.smart_topics == []
        assert sb._config.on_switch is None
        assert sb._config.log_decisions is False


class TestChatForwardsArgs:
    async def test_stream_is_real_llm_stream(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        ctx = _make_chat_ctx([("user", "hi")])
        stream = sb.chat(chat_ctx=ctx)
        assert isinstance(stream, llm.LLMStream)

    async def test_returns_fast_stream(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        ctx = _make_chat_ctx([("user", "hi")])
        sb.chat(chat_ctx=ctx)
        assert fast.chat_call_count == 1
        assert smart.chat_call_count == 0

    async def test_returns_smart_stream_after_escalation(self):
        fast, smart = _make_llm(), _make_llm()
        sb = Switchboard(fast=fast, smart=smart)
        ctx = _make_chat_ctx([("user", "No, that's wrong. Why? How?")])
        sb.chat(chat_ctx=ctx)
        assert smart.chat_call_count == 1
        assert fast.chat_call_count == 0
