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


def _analyze(message: str = "", *, topics: list[str] | None = None, **ctx_kwargs) -> Context:
    analyzer = HeuristicAnalyzer()
    config = SwitchboardConfig(smart_topics=topics or [])
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
        ctx = _analyze("I have a question about pricing", topics=["pricing", "warranty"])
        assert Signal.TOPIC_MATCH in ctx.signals_fired
        assert ctx.heuristic_score >= 0.50

    def test_no_topics_configured(self):
        ctx = _analyze("I have a question about pricing")
        assert Signal.TOPIC_MATCH not in ctx.signals_fired

    def test_topic_case_insensitive(self):
        ctx = _analyze("Tell me about WARRANTY options", topics=["warranty"])
        assert Signal.TOPIC_MATCH in ctx.signals_fired


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
        # pushback (0.40) + frustration (0.35) + topic (0.50) = 1.25 → 1.0
        ctx = _analyze(
            "No, that's wrong. I don't understand the pricing.",
            topics=["pricing"],
        )
        assert ctx.heuristic_score == 1.0
