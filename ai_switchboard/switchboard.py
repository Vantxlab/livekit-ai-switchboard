from __future__ import annotations

import logging
from typing import Any, Union

from livekit.agents import llm
from livekit.agents.llm import ChatContext, LLMStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr

from .analyzer import HeuristicAnalyzer
from .config import SwitchboardConfig
from .context import Context
from .events import SwitchEvent
from .rule import Rule

logger = logging.getLogger("ai_switchboard")


class Switchboard(llm.LLM):
    """Intelligent LLM router for LiveKit voice agents.

    Wraps two LLM instances (*fast* and *smart*) and transparently routes
    each ``chat()`` call to the best model based on heuristic signal
    analysis and optional developer-defined rules.
    """

    def __init__(
        self,
        *,
        fast: llm.LLM,
        smart: llm.LLM,
        config: SwitchboardConfig | None = None,
        rules: list[Rule] | None = None,
    ) -> None:
        super().__init__()
        self._fast = fast
        self._smart = smart
        self._config = config or SwitchboardConfig()
        self._rules = sorted(rules or [], key=lambda r: r.priority, reverse=True)
        self._analyzer = HeuristicAnalyzer()

        # Internal state
        self._current_model: str = self._config.start_on
        self._turn_count: int = 0
        self._turns_on_current: int = 0
        self._last_switch_turn: int = -1
        self._interruption_count: int = 0
        self._repeat_request_count: int = 0

    # -- Public helpers -------------------------------------------------------

    @property
    def current_model(self) -> str:
        """Return ``\"fast\"`` or ``\"smart\"``."""
        return self._current_model

    def record_interruption(self) -> None:
        """Call this when the user interrupts the agent mid-speech."""
        self._interruption_count += 1

    # -- llm.LLM interface ----------------------------------------------------

    @property
    def model(self) -> str:  # type: ignore[override]
        chosen = self._fast if self._current_model == "fast" else self._smart
        return chosen.model

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
        # 1. Extract last user message
        last_text = ""
        for msg in reversed(chat_ctx.messages):
            if msg.role == "user":
                last_text = msg.text_content or ""
                break

        # 2. Build context
        ctx = Context(
            last_message=last_text,
            last_message_word_count=len(last_text.split()),
            turn_count=self._turn_count,
            interruption_count=self._interruption_count,
            repeat_request_count=self._repeat_request_count,
            current_model=self._current_model,
            turns_on_current_model=self._turns_on_current,
            last_switch_turn=self._last_switch_turn,
        )

        # 3. Analyze signals
        self._analyzer.analyze(ctx, self._config)

        # Track repeat-request accumulation
        if "repeat_request" in ctx.signals_fired:
            self._repeat_request_count += 1

        # 4. Evaluate rules (highest priority first)
        target: str = self._current_model
        triggered_by: str = "heuristic"

        for rule in self._rules:
            if rule.condition(ctx):
                target = rule.use
                triggered_by = rule.name
                break
        else:
            # 5. No rule matched — apply heuristic thresholds
            if ctx.heuristic_score >= self._config.escalation_threshold:
                target = "smart"
            elif ctx.heuristic_score <= self._config.deescalation_threshold:
                target = "fast"

        # 6. Apply cooldown guard (don't de-escalate too soon)
        if (
            target == "fast"
            and self._current_model == "smart"
            and self._turns_on_current < self._config.cooldown_turns
        ):
            target = "smart"

        # 7. Determine if model actually changed
        changed = target != self._current_model

        # 8. Emit event
        event = SwitchEvent(
            turn=self._turn_count,
            from_model=self._current_model,
            to_model=target,
            triggered_by=triggered_by,
            signals_fired=list(ctx.signals_fired),
            heuristic_score=ctx.heuristic_score,
            changed=changed,
        )

        if self._config.on_switch is not None:
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

        # 9. Update internal state
        if changed:
            self._current_model = target
            self._turns_on_current = 0
            self._last_switch_turn = self._turn_count
        else:
            self._turns_on_current += 1

        self._turn_count += 1
        # Reset per-turn counters
        self._interruption_count = 0

        # 10. Forward to chosen LLM
        chosen = self._fast if self._current_model == "fast" else self._smart
        return chosen.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )
