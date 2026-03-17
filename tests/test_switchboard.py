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


def _make_models(names: list[str] | None = None) -> dict[str, MagicMock]:
    """Build a dict of named mock LLMs."""
    names = names or ["fast", "standard", "premium"]
    return {n: _make_mock_llm(f"{n}-model") for n in names}


def _make_chat_ctx(messages: list[tuple[str, str]] | None = None) -> MagicMock:
    """Build a fake ChatContext with a list of (role, text) tuples."""
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


def _chat(sb: Switchboard, text: str = "hello") -> Any:
    """Convenience wrapper to call ``sb.chat()`` with a simple user message."""
    ctx = _make_chat_ctx([("user", text)])
    return sb.chat(chat_ctx=ctx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicRouting:
    def test_starts_on_default(self):
        models = _make_models()
        sb = Switchboard(models=models)
        assert sb.current_model == "fast"

    def test_default_model_config(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(default_model="standard"),
        )
        assert sb.current_model == "standard"

    def test_simple_message_stays_on_default(self):
        models = _make_models()
        sb = Switchboard(models=models)
        _chat(sb, "Hi there!")
        assert sb.current_model == "fast"
        models["fast"].chat.assert_called_once()
        models["premium"].chat.assert_not_called()

    def test_provider_is_switchboard(self):
        models = _make_models()
        sb = Switchboard(models=models)
        assert sb.provider == "switchboard"

    def test_requires_at_least_two_models(self):
        import pytest

        with pytest.raises(ValueError, match="at least 2"):
            Switchboard(models={"solo": _make_mock_llm()})

    def test_list_of_tuples_input(self):
        fast, smart = _make_mock_llm("fast-m"), _make_mock_llm("smart-m")
        sb = Switchboard(models=[("fast", fast), ("smart", smart)])
        assert sb.current_model == "fast"
        assert sb.default_model == "fast"


class TestTopicRouting:
    def test_topic_routes_to_correct_model(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={
                    "premium": ["pricing", "billing"],
                    "standard": ["complaint", "support"],
                },
            ),
        )
        _chat(sb, "I have a question about pricing")
        assert sb.current_model == "premium"

    def test_topic_routes_to_standard(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={
                    "premium": ["pricing"],
                    "standard": ["complaint"],
                },
            ),
        )
        _chat(sb, "I have a complaint about the service")
        assert sb.current_model == "standard"

    def test_no_topic_match_stays_default(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
            ),
        )
        _chat(sb, "What's the weather?")
        assert sb.current_model == "fast"


class TestHeuristicEscalation:
    def test_escalation_model_triggered_by_high_score(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                escalation_model="premium",
                escalation_threshold=0.6,
            ),
        )
        # pushback (0.40) + multi_question (0.30) = 0.70 > 0.60
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb.current_model == "premium"

    def test_no_escalation_model_ignores_heuristic(self):
        models = _make_models()
        sb = Switchboard(models=models)
        # High score but no escalation_model configured
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb.current_model == "fast"

    def test_below_threshold_stays_default(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                escalation_model="premium",
                escalation_threshold=0.6,
            ),
        )
        _chat(sb, "Hello there")
        assert sb.current_model == "fast"


class TestHigherModelWins:
    def test_topic_standard_heuristic_premium(self):
        """When topic says 'standard' and heuristic says 'premium', premium wins."""
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"standard": ["complaint"]},
                escalation_model="premium",
                escalation_threshold=0.6,
            ),
        )
        # "complaint" matches standard topic, and pushback+multi_question triggers heuristic
        _chat(sb, "No, that's wrong. I have a complaint. Why? How?")
        assert sb.current_model == "premium"

    def test_topic_premium_heuristic_standard(self):
        """When topic says 'premium' and heuristic says 'standard', premium wins."""
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
                escalation_model="standard",
                escalation_threshold=0.6,
            ),
        )
        # "pricing" matches premium topic, heuristic would say standard
        _chat(sb, "No, that's wrong. Tell me about pricing. Why? How?")
        assert sb.current_model == "premium"


class TestCooldown:
    def test_hold_after_switch(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
                cooldown_turns=2,
            ),
        )
        # Escalate via topic
        _chat(sb, "Tell me about pricing")
        assert sb.current_model == "premium"
        # Simple message — cooldown should hold on premium
        _chat(sb, "ok")
        assert sb.current_model == "premium"
        # Still in cooldown (turns_on_current=1, need 2)
        _chat(sb, "yes")
        assert sb.current_model == "premium"
        # Now cooldown expired, should go back to default
        _chat(sb, "thanks")
        assert sb.current_model == "fast"

    def test_rules_bypass_cooldown(self):
        models = _make_models()
        rule = Rule(
            name="force_fast",
            condition=lambda ctx: "force-fast" in ctx.last_message,
            use="fast",
            priority=10,
        )
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
                cooldown_turns=99,
            ),
            rules=[rule],
        )
        # Escalate
        _chat(sb, "Tell me about pricing")
        assert sb.current_model == "premium"
        # Rule should bypass cooldown
        _chat(sb, "force-fast please")
        assert sb.current_model == "fast"


class TestRuleOverride:
    def test_rule_takes_priority(self):
        models = _make_models()
        rule = Rule(
            name="vip",
            condition=lambda ctx: "vip" in ctx.last_message.lower(),
            use="premium",
            priority=10,
        )
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"standard": ["vip"]},
            ),
            rules=[rule],
        )
        # Both topic (standard) and rule (premium) match — rule wins
        _chat(sb, "I am a VIP customer")
        assert sb.current_model == "premium"

    def test_higher_priority_rule_wins(self):
        models = _make_models()
        rules = [
            Rule(
                name="always_premium",
                condition=lambda ctx: True,
                use="premium",
                priority=1,
            ),
            Rule(
                name="always_fast", condition=lambda ctx: True, use="fast", priority=10
            ),
        ]
        sb = Switchboard(models=models, rules=rules)
        _chat(sb, "hello")
        assert sb.current_model == "fast"


class TestRuleErrorHandling:
    def test_broken_rule_does_not_crash(self):
        models = _make_models()
        rules = [
            Rule(
                name="broken",
                condition=lambda ctx: 1 / 0,  # ZeroDivisionError
                use="premium",
                priority=10,
            ),
            Rule(
                name="fallback",
                condition=lambda ctx: True,
                use="standard",
                priority=5,
            ),
        ]
        sb = Switchboard(models=models, rules=rules)
        _chat(sb, "hello")
        # Broken rule skipped, fallback rule should match
        assert sb.current_model == "standard"


class TestReset:
    def test_reset_clears_state(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
            ),
        )
        _chat(sb, "Tell me about pricing")
        assert sb.current_model == "premium"
        assert sb._turn_count == 1

        sb.reset()

        assert sb.current_model == "fast"
        assert sb._turn_count == 0
        assert sb._turns_on_current == 0
        assert sb._last_switch_turn == -1
        assert sb._interruption_count == 0
        assert sb._repeat_request_count == 0
        assert sb._audio_duration is None


class TestSTTConfidence:
    def test_transcript_confidence_flows_to_context(self):
        """stt_confidence from ChatMessage flows into the Context."""
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                escalation_model="premium",
                escalation_threshold=0.3,
            ),
        )
        # Build a chat context where the message has transcript_confidence
        chat_ctx = _make_chat_ctx([("user", "Hello")])
        chat_ctx.messages[0].transcript_confidence = 0.4
        sb.chat(chat_ctx=chat_ctx)
        # low_stt_confidence (0.30) >= 0.3 threshold -> escalate
        assert sb.current_model == "premium"


class TestAudioDuration:
    def test_audio_duration_flows_into_context(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                escalation_model="premium",
                escalation_threshold=0.1,
                long_audio_threshold=5.0,
            ),
        )
        sb.record_audio_duration(12.0)
        _chat(sb, "Hello")
        # long_audio_turn (0.15) >= 0.1 threshold -> escalate
        assert sb.current_model == "premium"

    def test_audio_duration_resets_after_turn(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                escalation_model="premium",
                escalation_threshold=0.1,
                long_audio_threshold=5.0,
            ),
        )
        sb.record_audio_duration(12.0)
        _chat(sb, "Hello")
        assert sb.current_model == "premium"
        # Audio duration should be reset, next turn should not see it
        # (but cooldown will hold, so we use rules to test)


class TestCallbacks:
    def test_on_switch_fires_only_on_change(self):
        switch_events: list[SwitchEvent] = []
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                on_switch=switch_events.append,
                model_topics={"premium": ["pricing"]},
            ),
        )
        _chat(sb, "Hi")  # no change
        assert len(switch_events) == 0

        _chat(sb, "Tell me about pricing")  # change!
        assert len(switch_events) == 1
        assert switch_events[0].changed is True

    def test_on_decision_fires_every_turn(self):
        decision_events: list[SwitchEvent] = []
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                on_decision=decision_events.append,
            ),
        )
        _chat(sb, "Hi")
        _chat(sb, "Hello again")
        assert len(decision_events) == 2
        assert decision_events[0].changed is False
        assert decision_events[1].changed is False

    def test_both_callbacks_fire_on_change(self):
        switch_events: list[SwitchEvent] = []
        decision_events: list[SwitchEvent] = []
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                on_switch=switch_events.append,
                on_decision=decision_events.append,
                model_topics={"premium": ["pricing"]},
            ),
        )
        _chat(sb, "Tell me about pricing")
        assert len(switch_events) == 1
        assert len(decision_events) == 1


class TestModelProperty:
    def test_model_reflects_current(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
            ),
        )
        assert sb.model == "fast-model"
        _chat(sb, "Tell me about pricing")
        assert sb.model == "premium-model"


class TestInterruption:
    def test_record_interruption(self):
        models = _make_models()
        sb = Switchboard(models=models)
        sb.record_interruption()
        # interruption (0.20) alone won't trigger escalation without escalation_model
        _chat(sb, "hello")
        assert sb.current_model == "fast"

    def test_interruption_resets_after_turn(self):
        decision_events: list[SwitchEvent] = []
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(on_decision=decision_events.append),
        )
        sb.record_interruption()
        _chat(sb, "hello")
        assert "interruption" in decision_events[0].signals_fired

        _chat(sb, "hi again")
        assert "interruption" not in decision_events[1].signals_fired


class TestValidation:
    def test_invalid_default_model(self):
        import pytest

        models = _make_models()
        with pytest.raises(ValueError, match="default_model"):
            Switchboard(
                models=models,
                config=SwitchboardConfig(default_model="nonexistent"),
            )

    def test_invalid_escalation_model(self):
        import pytest

        models = _make_models()
        with pytest.raises(ValueError, match="escalation_model"):
            Switchboard(
                models=models,
                config=SwitchboardConfig(escalation_model="nonexistent"),
            )

    def test_invalid_model_topics_key(self):
        import pytest

        models = _make_models()
        with pytest.raises(ValueError, match="model_topics"):
            Switchboard(
                models=models,
                config=SwitchboardConfig(model_topics={"nonexistent": ["test"]}),
            )

    def test_invalid_rule_use(self):
        import pytest

        models = _make_models()
        rule = Rule(name="bad", condition=lambda ctx: True, use="nonexistent")
        with pytest.raises(ValueError, match="Rule"):
            Switchboard(models=models, rules=[rule])

    def test_invalid_timeout_fallback_model(self):
        import pytest

        models = _make_models()
        with pytest.raises(ValueError, match="timeout_fallback_model"):
            Switchboard(
                models=models,
                config=SwitchboardConfig(timeout_fallback_model="nonexistent"),
            )


class TestMinSignalsForEscalation:
    """#7 — min_signals_for_escalation gates heuristic escalation."""

    def test_single_signal_blocked_when_min_is_two(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                escalation_model="premium",
                escalation_threshold=0.3,
                min_signals_for_escalation=2,
            ),
        )
        # pushback alone (0.40 >= 0.3 threshold) but only 1 signal
        _chat(sb, "No, that's wrong")
        assert sb.current_model == "fast"

    def test_two_signals_pass_when_min_is_two(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                escalation_model="premium",
                escalation_threshold=0.3,
                min_signals_for_escalation=2,
            ),
        )
        # pushback (0.40) + multi_question (0.30) = 0.70, 2 signals
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb.current_model == "premium"

    def test_default_min_is_one(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                escalation_model="premium",
                escalation_threshold=0.3,
            ),
        )
        # Single signal should escalate with default min=1
        _chat(sb, "No, that's wrong")
        assert sb.current_model == "premium"

    def test_topic_routing_bypasses_min_signals(self):
        """Topic routing is not gated by min_signals_for_escalation."""
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
                min_signals_for_escalation=99,
            ),
        )
        _chat(sb, "Tell me about pricing")
        assert sb.current_model == "premium"

    def test_rules_bypass_min_signals(self):
        models = _make_models()
        rule = Rule(
            name="force_premium",
            condition=lambda ctx: "vip" in ctx.last_message.lower(),
            use="premium",
            priority=10,
        )
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                min_signals_for_escalation=99,
            ),
            rules=[rule],
        )
        _chat(sb, "I am a VIP customer")
        assert sb.current_model == "premium"


class TestRecentMessages:
    """#2 — Sliding window of recent user messages."""

    def test_recent_messages_populated(self):
        events: list[SwitchEvent] = []
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                on_decision=events.append,
                context_window_size=3,
            ),
        )
        _chat(sb, "Hello")
        _chat(sb, "How are you")
        _chat(sb, "Fine thanks")
        assert list(sb._recent_messages) == ["Hello", "How are you", "Fine thanks"]

    def test_window_trims_old_messages(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(context_window_size=2),
        )
        _chat(sb, "First")
        _chat(sb, "Second")
        _chat(sb, "Third")
        assert list(sb._recent_messages) == ["Second", "Third"]

    def test_reset_clears_recent_messages(self):
        models = _make_models()
        sb = Switchboard(models=models)
        _chat(sb, "Hello")
        assert len(sb._recent_messages) > 0
        sb.reset()
        assert len(sb._recent_messages) == 0

    def test_rule_can_access_recent_messages(self):
        """Rules can use ctx.recent_messages to detect repetition."""
        models = _make_models()

        def repeated_question(ctx):
            if len(ctx.recent_messages) >= 2:
                return ctx.recent_messages[-1] == ctx.recent_messages[-2]
            return False

        rule = Rule(
            name="repeat_detector",
            condition=repeated_question,
            use="premium",
            priority=10,
        )
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(context_window_size=5),
            rules=[rule],
        )
        _chat(sb, "What is the price?")
        assert sb.current_model == "fast"
        _chat(sb, "What is the price?")
        assert sb.current_model == "premium"


class TestChatCtxInContext:
    """#8 — Rules can access chat_ctx via Context."""

    def test_chat_ctx_passed_to_rules(self):
        received_ctx = []

        def capture_rule(ctx):
            received_ctx.append(ctx)
            return False

        models = _make_models()
        rule = Rule(name="capture", condition=capture_rule, use="premium", priority=10)
        sb = Switchboard(models=models, rules=[rule])

        chat_ctx = _make_chat_ctx([("user", "Hello")])
        sb.chat(chat_ctx=chat_ctx)

        assert len(received_ctx) == 1
        assert received_ctx[0].chat_ctx is chat_ctx

    def test_rule_inspects_assistant_messages(self):
        """Rule can inspect previous assistant messages via chat_ctx."""
        models = _make_models()

        def after_tool_call(ctx):
            if ctx.chat_ctx is None:
                return False
            msgs = ctx.chat_ctx.messages
            for msg in msgs:
                if msg.role == "assistant" and "scheduled" in (
                    msg.text_content or ""
                ).lower():
                    return True
            return False

        rule = Rule(
            name="post_tool",
            condition=after_tool_call,
            use="fast",
            priority=10,
        )
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(model_topics={"premium": ["pricing"]}),
            rules=[rule],
        )
        # Chat context with assistant message containing "scheduled"
        chat_ctx = _make_chat_ctx(
            [
                ("user", "Schedule my appointment"),
                ("assistant", "I've scheduled your appointment for 3pm"),
                ("user", "Tell me about pricing"),
            ]
        )
        sb.chat(chat_ctx=chat_ctx)
        # Rule fires and forces fast, even though topic would pick premium
        assert sb.current_model == "fast"


class TestLatencyAwareRouting:
    """#3 — TTFB tracking and auto-demotion."""

    def test_no_ttfb_data_no_demotion(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
                max_ttfb_ms={"premium": 500.0},
                timeout_fallback_model="fast",
            ),
        )
        # No TTFB recorded — should route normally
        _chat(sb, "Tell me about pricing")
        assert sb.current_model == "premium"

    def test_low_ttfb_no_demotion(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
                max_ttfb_ms={"premium": 500.0},
                timeout_fallback_model="fast",
            ),
        )
        # Record fast TTFB
        for _ in range(5):
            sb.record_ttfb("premium", 200.0)
        _chat(sb, "Tell me about pricing")
        assert sb.current_model == "premium"

    def test_high_ttfb_demotes_to_fallback(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
                max_ttfb_ms={"premium": 500.0},
                timeout_fallback_model="fast",
            ),
        )
        # Record slow TTFB
        for _ in range(5):
            sb.record_ttfb("premium", 800.0)
        _chat(sb, "Tell me about pricing")
        assert sb.current_model == "fast"

    def test_ttfb_rolling_window(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
                max_ttfb_ms={"premium": 500.0},
                timeout_fallback_model="fast",
                ttfb_window_size=3,
            ),
        )
        # Old slow samples
        sb.record_ttfb("premium", 900.0)
        sb.record_ttfb("premium", 900.0)
        sb.record_ttfb("premium", 900.0)
        # New fast samples push old ones out of the window
        sb.record_ttfb("premium", 100.0)
        sb.record_ttfb("premium", 100.0)
        sb.record_ttfb("premium", 100.0)
        _chat(sb, "Tell me about pricing")
        # avg is 100.0, under threshold
        assert sb.current_model == "premium"

    def test_rules_bypass_ttfb_demotion(self):
        models = _make_models()
        rule = Rule(
            name="force_premium",
            condition=lambda ctx: "vip" in ctx.last_message.lower(),
            use="premium",
            priority=10,
        )
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                max_ttfb_ms={"premium": 500.0},
                timeout_fallback_model="fast",
            ),
            rules=[rule],
        )
        for _ in range(5):
            sb.record_ttfb("premium", 800.0)
        _chat(sb, "I am a VIP")
        assert sb.current_model == "premium"

    def test_ttfb_event_triggered_by(self):
        events: list[SwitchEvent] = []
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
                max_ttfb_ms={"premium": 500.0},
                timeout_fallback_model="fast",
                on_decision=events.append,
            ),
        )
        for _ in range(5):
            sb.record_ttfb("premium", 800.0)
        _chat(sb, "Tell me about pricing")
        assert events[0].triggered_by == "ttfb_fallback"

    def test_reset_clears_ttfb(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                model_topics={"premium": ["pricing"]},
                max_ttfb_ms={"premium": 500.0},
                timeout_fallback_model="fast",
            ),
        )
        for _ in range(5):
            sb.record_ttfb("premium", 800.0)
        sb.reset()
        # After reset, no TTFB data — should route normally
        _chat(sb, "Tell me about pricing")
        assert sb.current_model == "premium"


class TestConfigurableWeightsRouting:
    """#4 — Signal weights affect routing decisions end-to-end."""

    def test_lowered_weight_prevents_escalation(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                escalation_model="premium",
                escalation_threshold=0.6,
                signal_weights={"pushback": 0.05, "multi_question": 0.05},
            ),
        )
        # pushback + multi_question normally = 0.70, but overridden to 0.10
        _chat(sb, "No, that's wrong. Why? How?")
        assert sb.current_model == "fast"

    def test_raised_weight_triggers_escalation(self):
        models = _make_models()
        sb = Switchboard(
            models=models,
            config=SwitchboardConfig(
                escalation_model="premium",
                escalation_threshold=0.6,
                signal_weights={"frustration": 0.70},
            ),
        )
        # frustration alone with weight 0.70 >= 0.6 threshold
        _chat(sb, "I don't understand what you're saying")
        assert sb.current_model == "premium"
