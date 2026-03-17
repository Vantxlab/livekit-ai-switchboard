from ai_switchboard.context import Context
from ai_switchboard.rule import Rule


def _make_ctx(**kwargs) -> Context:
    defaults = dict(
        last_message="hello",
        last_message_word_count=1,
        current_model="fast",
    )
    defaults.update(kwargs)
    return Context(**defaults)


class TestRuleDataclass:
    def test_defaults(self):
        rule = Rule(name="test", condition=lambda ctx: True, use="smart")
        assert rule.priority == 5

    def test_custom_priority(self):
        rule = Rule(name="test", condition=lambda ctx: True, use="smart", priority=10)
        assert rule.priority == 10

    def test_use_accepts_any_string(self):
        rule = Rule(name="test", condition=lambda ctx: True, use="premium")
        assert rule.use == "premium"

        rule2 = Rule(name="test2", condition=lambda ctx: True, use="my-custom-model")
        assert rule2.use == "my-custom-model"


class TestRuleSorting:
    def test_sorted_by_priority_descending(self):
        rules = [
            Rule(name="low", condition=lambda ctx: True, use="smart", priority=1),
            Rule(name="high", condition=lambda ctx: True, use="fast", priority=10),
            Rule(name="mid", condition=lambda ctx: True, use="smart", priority=5),
        ]
        sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        assert [r.name for r in sorted_rules] == ["high", "mid", "low"]


class TestRuleConditions:
    def test_condition_receives_context(self):
        received = []

        def capture(ctx: Context) -> bool:
            received.append(ctx)
            return True

        rule = Rule(name="capture", condition=capture, use="smart")
        ctx = _make_ctx(last_message="test message")
        result = rule.condition(ctx)

        assert result is True
        assert len(received) == 1
        assert received[0].last_message == "test message"

    def test_condition_based_on_word_count(self):
        rule = Rule(
            name="short_message",
            condition=lambda ctx: ctx.last_message_word_count < 5,
            use="fast",
        )
        short_ctx = _make_ctx(last_message="hi", last_message_word_count=1)
        long_ctx = _make_ctx(last_message="a b c d e f", last_message_word_count=6)

        assert rule.condition(short_ctx) is True
        assert rule.condition(long_ctx) is False

    def test_first_matching_rule_wins(self):
        rules = [
            Rule(name="high", condition=lambda ctx: True, use="fast", priority=10),
            Rule(name="low", condition=lambda ctx: True, use="smart", priority=1),
        ]
        sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)

        winner = None
        for rule in sorted_rules:
            if rule.condition(_make_ctx()):
                winner = rule
                break

        assert winner is not None
        assert winner.name == "high"
        assert winner.use == "fast"
