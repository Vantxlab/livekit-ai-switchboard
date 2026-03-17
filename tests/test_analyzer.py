from __future__ import annotations

from ai_switchboard.analyzer import HeuristicAnalyzer
from ai_switchboard.config import SwitchboardConfig
from ai_switchboard.context import Context
from ai_switchboard.signal import Signal


def _make_ctx(message: str = "", **kwargs) -> Context:
    return Context(
        last_message=message,
        last_message_word_count=len(message.split()) if message else 0,
        **kwargs,
    )


def _analyze(
    message: str = "",
    *,
    model_topics: dict[str, list[str]] | None = None,
    stt_confidence_threshold: float = 0.7,
    long_audio_threshold: float = 10.0,
    **ctx_kwargs,
) -> Context:
    analyzer = HeuristicAnalyzer()
    config = SwitchboardConfig(
        model_topics=model_topics or {},
        stt_confidence_threshold=stt_confidence_threshold,
        long_audio_threshold=long_audio_threshold,
    )
    ctx = _make_ctx(message, **ctx_kwargs)
    return analyzer.analyze(ctx, config)


class TestLongInput:
    def test_short_message_no_signal(self):
        ctx = _analyze("Hello there")
        assert Signal.LONG_INPUT not in ctx.signals_fired

    def test_long_message_fires(self):
        words = " ".join(["word"] * 26)
        ctx = _analyze(words)
        assert Signal.LONG_INPUT in ctx.signals_fired
        assert ctx.heuristic_score >= 0.20


class TestMultiQuestion:
    def test_single_question_no_signal(self):
        ctx = _analyze("How are you?")
        assert Signal.MULTI_QUESTION not in ctx.signals_fired

    def test_two_questions_fires(self):
        ctx = _analyze("How are you? What do you think?")
        assert Signal.MULTI_QUESTION in ctx.signals_fired

    def test_three_questions(self):
        ctx = _analyze("Who? What? Where?")
        assert Signal.MULTI_QUESTION in ctx.signals_fired


class TestComplexityWords:
    def test_explain_fires(self):
        ctx = _analyze("Can you explain how this works?")
        assert Signal.COMPLEXITY_WORDS in ctx.signals_fired

    def test_compare_fires(self):
        ctx = _analyze("Compare these two options")
        assert Signal.COMPLEXITY_WORDS in ctx.signals_fired

    def test_no_match(self):
        ctx = _analyze("Hello, I want to order a pizza")
        assert Signal.COMPLEXITY_WORDS not in ctx.signals_fired


class TestPushback:
    def test_thats_wrong_fires(self):
        ctx = _analyze("No, that's wrong, I said something else")
        assert Signal.PUSHBACK in ctx.signals_fired

    def test_incorrect_fires(self):
        ctx = _analyze("That's incorrect")
        assert Signal.PUSHBACK in ctx.signals_fired

    def test_no_match(self):
        ctx = _analyze("Yes, that sounds good")
        assert Signal.PUSHBACK not in ctx.signals_fired


class TestFrustration:
    def test_doesnt_make_sense(self):
        ctx = _analyze("This doesn't make sense at all")
        assert Signal.FRUSTRATION in ctx.signals_fired

    def test_confused(self):
        ctx = _analyze("I'm confused by what you said")
        assert Signal.FRUSTRATION in ctx.signals_fired


class TestRepeatRequest:
    def test_say_that_again(self):
        ctx = _analyze("Can you say that again please?")
        assert Signal.REPEAT_REQUEST in ctx.signals_fired

    def test_pardon(self):
        ctx = _analyze("Pardon?")
        assert Signal.REPEAT_REQUEST in ctx.signals_fired


class TestInterruption:
    def test_no_interruptions(self):
        ctx = _analyze("Hello", interruption_count=0)
        assert Signal.INTERRUPTION not in ctx.signals_fired

    def test_with_interruptions(self):
        ctx = _analyze("Hello", interruption_count=1)
        assert Signal.INTERRUPTION in ctx.signals_fired


class TestTopicMatch:
    def test_topic_hit(self):
        ctx = _analyze(
            "I have a question about pricing",
            model_topics={"premium": ["pricing", "warranty"]},
        )
        assert Signal.TOPIC_MATCH in ctx.signals_fired
        assert ctx.heuristic_score >= 0.50

    def test_no_topics_configured(self):
        ctx = _analyze("I have a question about pricing")
        assert Signal.TOPIC_MATCH not in ctx.signals_fired

    def test_topic_case_insensitive(self):
        ctx = _analyze(
            "Tell me about WARRANTY options",
            model_topics={"premium": ["warranty"]},
        )
        assert Signal.TOPIC_MATCH in ctx.signals_fired


class TestLowSTTConfidence:
    def test_none_confidence_no_signal(self):
        ctx = _analyze("Hello", stt_confidence=None)
        assert Signal.LOW_STT_CONFIDENCE not in ctx.signals_fired

    def test_high_confidence_no_signal(self):
        ctx = _analyze("Hello", stt_confidence=0.95)
        assert Signal.LOW_STT_CONFIDENCE not in ctx.signals_fired

    def test_low_confidence_fires(self):
        ctx = _analyze("Hello", stt_confidence=0.5)
        assert Signal.LOW_STT_CONFIDENCE in ctx.signals_fired
        assert ctx.heuristic_score >= 0.30

    def test_custom_threshold(self):
        ctx = _analyze("Hello", stt_confidence=0.65, stt_confidence_threshold=0.5)
        assert Signal.LOW_STT_CONFIDENCE not in ctx.signals_fired


class TestLongAudioTurn:
    def test_none_duration_no_signal(self):
        ctx = _analyze("Hello", audio_duration=None)
        assert Signal.LONG_AUDIO_TURN not in ctx.signals_fired

    def test_short_duration_no_signal(self):
        ctx = _analyze("Hello", audio_duration=5.0)
        assert Signal.LONG_AUDIO_TURN not in ctx.signals_fired

    def test_long_duration_fires(self):
        ctx = _analyze("Hello", audio_duration=15.0)
        assert Signal.LONG_AUDIO_TURN in ctx.signals_fired
        assert ctx.heuristic_score >= 0.15

    def test_custom_threshold(self):
        ctx = _analyze("Hello", audio_duration=8.0, long_audio_threshold=5.0)
        assert Signal.LONG_AUDIO_TURN in ctx.signals_fired


class TestSubstringFix:
    def test_know_does_not_trigger_pushback(self):
        """'I know what you mean' must NOT trigger pushback from 'no' substring."""
        ctx = _analyze("I know what you mean")
        assert Signal.PUSHBACK not in ctx.signals_fired

    def test_no_thats_wrong_triggers_pushback(self):
        """'No, that's wrong' must trigger pushback."""
        ctx = _analyze("No, that's wrong")
        assert Signal.PUSHBACK in ctx.signals_fired

    def test_acknowledge_does_not_trigger_pushback(self):
        """Words containing 'no' as substring should not trigger."""
        ctx = _analyze("I want to innovate and explore new technology")
        assert Signal.PUSHBACK not in ctx.signals_fired

    def test_standalone_no_triggers(self):
        ctx = _analyze("No")
        assert Signal.PUSHBACK in ctx.signals_fired


class TestMultiWordBoundary:
    """#1 — Multi-word phrases use word boundaries, not bare substring."""

    def test_walk_me_through_fires(self):
        ctx = _analyze("Can you walk me through the process?")
        assert Signal.COMPLEXITY_WORDS in ctx.signals_fired

    def test_walk_me_through_not_embedded(self):
        """'sidewalk me through' should not trigger (word boundary)."""
        ctx = _analyze("The sidewalk met hrough the park")
        assert Signal.COMPLEXITY_WORDS not in ctx.signals_fired

    def test_what_if_fires(self):
        ctx = _analyze("What if we try a different approach?")
        assert Signal.COMPLEXITY_WORDS in ctx.signals_fired

    def test_what_if_extra_whitespace(self):
        """Multi-word phrase with extra whitespace should still match."""
        ctx = _analyze("What  if we do it differently?")
        assert Signal.COMPLEXITY_WORDS in ctx.signals_fired

    def test_how_does_fires(self):
        ctx = _analyze("How does this work?")
        assert Signal.COMPLEXITY_WORDS in ctx.signals_fired

    def test_not_what_i_fires(self):
        ctx = _analyze("That's not what I asked for")
        assert Signal.PUSHBACK in ctx.signals_fired

    def test_say_that_again_fires(self):
        ctx = _analyze("Can you say that again?")
        assert Signal.REPEAT_REQUEST in ctx.signals_fired

    def test_i_already_fires(self):
        ctx = _analyze("I already told you my name")
        assert Signal.PUSHBACK in ctx.signals_fired


class TestTopicWordBoundary:
    """#1 — Topic matching uses word boundaries."""

    def test_topic_exact_word_fires(self):
        ctx = _analyze(
            "Tell me about pricing",
            model_topics={"premium": ["pricing"]},
        )
        assert Signal.TOPIC_MATCH in ctx.signals_fired

    def test_topic_substring_does_not_fire(self):
        """'repricing' should not match topic 'pricing'."""
        ctx = _analyze(
            "The repricing strategy is interesting",
            model_topics={"premium": ["pricing"]},
        )
        assert Signal.TOPIC_MATCH not in ctx.signals_fired

    def test_multi_word_topic_fires(self):
        ctx = _analyze(
            "I need help with my credit card",
            model_topics={"premium": ["credit card"]},
        )
        assert Signal.TOPIC_MATCH in ctx.signals_fired

    def test_multi_word_topic_substring_no_fire(self):
        """'discredit card' should not match 'credit card'."""
        ctx = _analyze(
            "They discredit card companies often",
            model_topics={"premium": ["credit card"]},
        )
        assert Signal.TOPIC_MATCH not in ctx.signals_fired


class TestPrecompiledRegex:
    """#6 — Verify patterns are pre-compiled at init, not rebuilt per call."""

    def test_analyzer_has_compiled_patterns(self):
        from ai_switchboard.analyzer import HeuristicAnalyzer

        analyzer = HeuristicAnalyzer()
        assert hasattr(analyzer, "_compiled")
        assert Signal.PUSHBACK in analyzer._compiled
        assert Signal.FRUSTRATION in analyzer._compiled
        assert Signal.COMPLEXITY_WORDS in analyzer._compiled
        assert Signal.REPEAT_REQUEST in analyzer._compiled

    def test_compiled_pattern_is_reused(self):
        """Same analyzer instance reuses compiled patterns across calls."""
        import re

        from ai_switchboard.analyzer import HeuristicAnalyzer

        analyzer = HeuristicAnalyzer()
        pattern = analyzer._compiled[Signal.PUSHBACK]
        assert isinstance(pattern, re.Pattern)
        # Call analyze twice — pattern object identity should be stable
        config = SwitchboardConfig()
        ctx1 = _make_ctx("No, that's wrong")
        analyzer.analyze(ctx1, config)
        assert analyzer._compiled[Signal.PUSHBACK] is pattern


class TestConfigurableWeights:
    """#4 — Signal weights can be overridden via SwitchboardConfig."""

    def test_default_weights_unchanged(self):
        ctx = _analyze("No, that's wrong")
        # Default pushback weight is 0.40
        assert ctx.heuristic_score >= 0.40

    def test_custom_weight_override(self):
        analyzer = HeuristicAnalyzer()
        config = SwitchboardConfig(
            signal_weights={Signal.PUSHBACK: 0.10},
        )
        ctx = _make_ctx("No, that's wrong")
        analyzer.analyze(ctx, config)
        # With pushback weight lowered to 0.10
        assert 0.09 <= ctx.heuristic_score <= 0.11

    def test_custom_weight_for_new_signal(self):
        """Custom weights can add new signal keys (for future extensibility)."""
        analyzer = HeuristicAnalyzer()
        config = SwitchboardConfig(
            signal_weights={"custom_signal": 0.50},
        )
        ctx = _make_ctx("Hello")
        analyzer.analyze(ctx, config)
        # No custom signal detected, so score is still 0
        assert ctx.heuristic_score == 0.0

    def test_partial_override_preserves_defaults(self):
        """Overriding one weight preserves all others."""
        analyzer = HeuristicAnalyzer()
        config = SwitchboardConfig(
            signal_weights={Signal.PUSHBACK: 0.10},
        )
        # frustration (default 0.35) + pushback (overridden 0.10) = 0.45
        ctx = _make_ctx("No, that's wrong. I don't understand.")
        analyzer.analyze(ctx, config)
        assert 0.44 <= ctx.heuristic_score <= 0.46


class TestScoring:
    def test_empty_message_zero_score(self):
        ctx = _analyze("")
        assert ctx.heuristic_score == 0.0
        assert ctx.signals_fired == []

    def test_combined_signals(self):
        # pushback (0.40) + multi_question (0.30) = 0.70
        ctx = _analyze("No, that's wrong. Why? How?")
        assert Signal.PUSHBACK in ctx.signals_fired
        assert Signal.MULTI_QUESTION in ctx.signals_fired
        assert ctx.heuristic_score >= 0.70

    def test_score_capped_at_one(self):
        # pushback (0.40) + frustration (0.35) + topic (0.50) = 1.25 -> 1.0
        ctx = _analyze(
            "No, that's wrong. I don't understand the pricing.",
            model_topics={"premium": ["pricing"]},
        )
        assert ctx.heuristic_score == 1.0
