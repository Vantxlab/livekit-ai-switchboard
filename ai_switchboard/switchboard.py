from __future__ import annotations

import logging
from collections import deque
from typing import Any, Union

from livekit.agents import llm
from livekit.agents.llm import ChatContext, LLMStream
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .analyzer import HeuristicAnalyzer
from .config import SwitchboardConfig
from .context import Context
from .events import SwitchEvent
from .metrics import SwitchboardMetrics
from .rule import Rule

logger = logging.getLogger("ai_switchboard")


class Switchboard(llm.LLM):
    """Intelligent LLM router for LiveKit voice agents.

    Routes each ``chat()`` call to the best model based on heuristic signal
    analysis, topic matching, and optional developer-defined rules.
    """

    def __init__(
        self,
        *,
        models: dict[str, llm.LLM] | list[tuple[str, llm.LLM]],
        config: SwitchboardConfig | None = None,
        rules: list[Rule] | None = None,
    ) -> None:
        super().__init__()

        # Normalize to dict (preserving insertion order)
        if isinstance(models, list):
            self._models: dict[str, llm.LLM] = dict(models)
        else:
            self._models = dict(models)

        if len(self._models) < 2:
            raise ValueError("Switchboard requires at least 2 models")

        self._model_order: list[str] = list(self._models.keys())
        self._config = config or SwitchboardConfig()
        self._rules = sorted(rules or [], key=lambda r: r.priority, reverse=True)
        self._analyzer = HeuristicAnalyzer()

        # Resolve default_model
        self._default_model: str = (
            self._config.default_model
            if self._config.default_model
            else self._model_order[0]
        )

        # Validate model references
        self._validate()

        # Internal state
        self._current_model: str = self._default_model
        self._turn_count: int = 0
        self._turns_on_current: int = 0
        self._last_switch_turn: int = -1
        self._interruption_count: int = 0
        self._repeat_request_count: int = 0
        self._audio_duration: float | None = None
        self._recent_messages: deque[str] = deque(
            maxlen=self._config.context_window_size
        )

        # Latency tracking: model_name -> rolling TTFB samples
        self._model_ttfb: dict[str, deque[float]] = {}

        # Metrics
        self._metrics = SwitchboardMetrics()

    def _validate(self) -> None:
        """Validate that all model references are valid."""
        valid = set(self._model_order)

        if self._default_model not in valid:
            raise ValueError(
                f"default_model {self._default_model!r} not in models: {self._model_order}"
            )

        if self._config.escalation_model and self._config.escalation_model not in valid:
            raise ValueError(
                f"escalation_model {self._config.escalation_model!r} not in models: {self._model_order}"
            )

        if (
            self._config.timeout_fallback_model
            and self._config.timeout_fallback_model not in valid
        ):
            raise ValueError(
                f"timeout_fallback_model {self._config.timeout_fallback_model!r} not in models: {self._model_order}"
            )

        for model_name in self._config.model_topics:
            if model_name not in valid:
                raise ValueError(
                    f"model_topics key {model_name!r} not in models: {self._model_order}"
                )

        for rule in self._rules:
            if rule.use not in valid:
                raise ValueError(
                    f"Rule {rule.name!r} targets model {rule.use!r} not in models: {self._model_order}"
                )

    # -- Public helpers -------------------------------------------------------

    @property
    def current_model(self) -> str:
        """Return the name of the currently active model."""
        return self._current_model

    @property
    def default_model(self) -> str:
        """Return the name of the default model."""
        return self._default_model

    @property
    def metrics(self) -> SwitchboardMetrics:
        """Return the metrics collector (read-only view)."""
        return self._metrics

    def record_interruption(self) -> None:
        """Call this when the user interrupts the agent mid-speech."""
        self._interruption_count += 1

    def record_audio_duration(self, seconds: float) -> None:
        """Record the audio duration of the current user turn."""
        self._audio_duration = seconds

    def record_ttfb(self, model: str, ms: float) -> None:
        """Record time-to-first-byte for a model (milliseconds)."""
        if model not in self._model_ttfb:
            self._model_ttfb[model] = deque(
                maxlen=self._config.ttfb_window_size
            )
        self._model_ttfb[model].append(ms)

    def _avg_ttfb(self, model: str) -> float | None:
        """Return rolling average TTFB for a model, or None if no data."""
        samples = self._model_ttfb.get(model)
        if not samples:
            return None
        return sum(samples) / len(samples)

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._current_model = self._default_model
        self._turn_count = 0
        self._turns_on_current = 0
        self._last_switch_turn = -1
        self._interruption_count = 0
        self._repeat_request_count = 0
        self._audio_duration = None
        self._recent_messages.clear()
        self._model_ttfb.clear()
        self._metrics.reset()

    def _model_tier(self, name: str) -> int:
        """Return the tier index of a model (higher = more capable)."""
        return self._model_order.index(name)

    # -- llm.LLM interface ----------------------------------------------------

    @property
    def model(self) -> str:  # type: ignore[override]
        return self._models[self._current_model].model

    @property
    def provider(self) -> str:  # type: ignore[override]
        return "switchboard"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: Union[list[llm.Tool], None] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        # 1. Extract last user message + stt_confidence
        last_text = ""
        stt_confidence = None
        try:
            items = chat_ctx.items
            if not isinstance(items, list):
                raise TypeError
        except (AttributeError, TypeError):
            items = chat_ctx.messages
        for msg in reversed(items):
            if msg.role == "user":
                last_text = msg.text_content or ""
                stt_confidence = getattr(msg, "transcript_confidence", None)
                break

        # 2. Update recent messages window
        if last_text:
            self._recent_messages.append(last_text)

        # 3. Build context
        ctx = Context(
            last_message=last_text,
            last_message_word_count=len(last_text.split()),
            turn_count=self._turn_count,
            recent_messages=list(self._recent_messages),
            interruption_count=self._interruption_count,
            repeat_request_count=self._repeat_request_count,
            current_model=self._current_model,
            turns_on_current_model=self._turns_on_current,
            last_switch_turn=self._last_switch_turn,
            stt_confidence=stt_confidence,
            audio_duration=self._audio_duration,
            chat_ctx=chat_ctx,
        )

        # 4. Analyze signals
        self._analyzer.analyze(ctx, self._config)

        # 5. Track repeat-request accumulation
        if "repeat_request" in ctx.signals_fired:
            self._repeat_request_count += 1

        # 6. Evaluate rules (highest priority first, first match wins)
        target: str | None = None
        triggered_by: str = "default"
        rule_matched = False

        for rule in self._rules:
            try:
                matched = rule.condition(ctx)
            except Exception:
                logger.warning(
                    "Rule %r raised an exception, skipping", rule.name, exc_info=True
                )
                continue
            if matched:
                target = rule.use
                triggered_by = rule.name
                rule_matched = True
                break

        if not rule_matched:
            # 7. Evaluate topic match: find the highest-tier model with a topic hit
            topic_target: str | None = None
            message_lower = ctx.last_message.lower()
            for model_name, topics in self._config.model_topics.items():
                if HeuristicAnalyzer._match_topic(message_lower, topics):
                    if topic_target is None or self._model_tier(
                        model_name
                    ) > self._model_tier(topic_target):
                        topic_target = model_name

            # 8. Evaluate heuristic escalation (gated by min_signals_for_escalation)
            heuristic_target: str | None = None
            if (
                self._config.escalation_model
                and ctx.heuristic_score >= self._config.escalation_threshold
                and len(ctx.signals_fired)
                >= self._config.min_signals_for_escalation
            ):
                heuristic_target = self._config.escalation_model

            # 9. Resolve: pick the HIGHER model between topic and heuristic
            candidates = [c for c in (topic_target, heuristic_target) if c is not None]
            if candidates:
                target = max(candidates, key=lambda n: self._model_tier(n))
                if target == topic_target and target == heuristic_target:
                    triggered_by = "topic+heuristic"
                elif target == topic_target:
                    triggered_by = "topic"
                else:
                    triggered_by = "heuristic"
            else:
                target = self._default_model
                triggered_by = "default"

        # 10. Apply cooldown: if target would de-escalate and we haven't served
        #     enough turns on current model, hold current. Rules bypass cooldown.
        if (
            not rule_matched
            and self._model_tier(target) < self._model_tier(self._current_model)
            and self._turns_on_current < self._config.cooldown_turns
        ):
            target = self._current_model

        # 11. Apply latency-aware routing: if chosen model's avg TTFB exceeds
        #     threshold, demote to fallback model. Rules bypass this.
        if (
            not rule_matched
            and self._config.max_ttfb_ms
            and self._config.timeout_fallback_model
            and target in self._config.max_ttfb_ms
        ):
            avg = self._avg_ttfb(target)
            if avg is not None and avg > self._config.max_ttfb_ms[target]:
                target = self._config.timeout_fallback_model
                triggered_by = "ttfb_fallback"

        # 12. Determine if changed, emit callbacks
        changed = target != self._current_model

        event = SwitchEvent(
            turn=self._turn_count,
            from_model=self._current_model,
            to_model=target,
            triggered_by=triggered_by,
            signals_fired=list(ctx.signals_fired),
            heuristic_score=ctx.heuristic_score,
            changed=changed,
        )

        if self._config.on_decision is not None:
            self._config.on_decision(event)

        if changed and self._config.on_switch is not None:
            self._config.on_switch(event)

        if self._config.log_decisions:
            logger.info(
                "turn=%d model=%s score=%.2f signals=%s triggered_by=%s changed=%s",
                self._turn_count,
                target,
                ctx.heuristic_score,
                ctx.signals_fired,
                triggered_by,
                changed,
            )

        # 13. Record metrics
        self._metrics.record_turn(
            model=target,
            triggered_by=triggered_by,
            heuristic_score=ctx.heuristic_score,
            changed=changed,
        )

        # 14. Update internal state
        if changed:
            self._current_model = target
            self._turns_on_current = 0
            self._last_switch_turn = self._turn_count
        else:
            self._turns_on_current += 1

        self._turn_count += 1
        # Reset per-turn counters
        self._interruption_count = 0
        self._audio_duration = None

        # 15. Forward to chosen LLM
        return self._models[self._current_model].chat(
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )
