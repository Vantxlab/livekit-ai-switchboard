"""Tests using LiveKit's AgentSession testing framework.

Uses a lightweight FakeLLM so tests run locally without API keys.
Validates N-model routing scenarios:
  - simple messages stay on fast (default)
  - "pricing" topic escalates to smart
  - interruptions escalate to smart via rule
  - de-escalation after cooldown
"""

from __future__ import annotations

import pytest
from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChoiceDelta
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice import Agent, AgentSession

from ai_switchboard import Rule, Switchboard, SwitchboardConfig, SwitchEvent

INSTRUCTIONS = "You are a helpful assistant. Keep responses very brief."


# ---------------------------------------------------------------------------
# FakeLLM — minimal llm.LLM that works with AgentSession without API keys
# ---------------------------------------------------------------------------


class FakeLLMStream(llm.LLMStream):
    """Emits a single canned response chunk then closes."""

    def __init__(self, *, llm_instance: llm.LLM, chat_ctx: llm.ChatContext, response: str) -> None:
        super().__init__(
            llm_instance,
            chat_ctx=chat_ctx,
            tools=[],
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )
        self._response = response

    async def _run(self) -> None:
        self._event_ch.send_nowait(
            ChatChunk(
                id="fake-0",
                delta=ChoiceDelta(role="assistant", content=self._response),
            )
        )


class FakeLLM(llm.LLM):
    """A fake LLM that returns a fixed response. No API keys needed."""

    def __init__(self, *, model_name: str = "fake-model", response: str = "OK") -> None:
        super().__init__()
        self._model_name = model_name
        self._response = response

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "fake"

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kwargs) -> llm.LLMStream:
        return FakeLLMStream(llm_instance=self, chat_ctx=chat_ctx, response=self._response)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_llm():
    return FakeLLM(model_name="groq/llama-3.3-70b-versatile", response="Sure, I can help!")


@pytest.fixture
def smart_llm():
    return FakeLLM(model_name="anthropic/claude-sonnet-4-6", response="Let me provide a detailed answer.")


@pytest.fixture
def switchboard(fast_llm, smart_llm):
    """Switchboard matching the openrouter_anthropic example config."""
    return Switchboard(
        models={"fast": fast_llm, "smart": smart_llm},
        config=SwitchboardConfig(
            model_topics={"smart": ["pricing"]},
            escalation_model="smart",
            escalation_threshold=0.5,
            cooldown_turns=2,
        ),
        rules=[
            Rule(
                name="interruption_escalation",
                condition=lambda ctx: ctx.interruption_count > 0,
                use="smart",
                priority=10,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests — LiveKit AgentSession framework
# ---------------------------------------------------------------------------


class TestAgentSessionOpenRouterAnthropic:
    """End-to-end routing tests using LiveKit's AgentSession."""

    async def test_simple_message_stays_on_fast(self, switchboard):
        async with AgentSession() as session:
            await session.start(Agent(instructions=INSTRUCTIONS, llm=switchboard))

            result = await session.run(user_input="Hi there!")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()

            assert switchboard.current_model == "fast"
            assert switchboard.model == "groq/llama-3.3-70b-versatile"

    async def test_pricing_escalates_to_smart(self, switchboard):
        async with AgentSession() as session:
            await session.start(Agent(instructions=INSTRUCTIONS, llm=switchboard))

            result = await session.run(user_input="What is the pricing for your enterprise plan?")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()

            assert switchboard.current_model == "smart"
            assert switchboard.model == "anthropic/claude-sonnet-4-6"

    async def test_interruption_escalates_to_smart(self, switchboard):
        async with AgentSession() as session:
            await session.start(Agent(instructions=INSTRUCTIONS, llm=switchboard))

            # Record an interruption before the next turn
            switchboard.record_interruption()

            result = await session.run(user_input="Wait, stop.")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()

            assert switchboard.current_model == "smart"

    async def test_deescalation_after_cooldown(self, switchboard):
        async with AgentSession() as session:
            await session.start(Agent(instructions=INSTRUCTIONS, llm=switchboard))

            # Escalate via pricing topic
            result = await session.run(user_input="Tell me about pricing")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert switchboard.current_model == "smart"

            # Cooldown turn 1
            result = await session.run(user_input="ok")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert switchboard.current_model == "smart"  # still in cooldown

            # Cooldown turn 2
            result = await session.run(user_input="got it")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert switchboard.current_model == "smart"  # still in cooldown

            # Now should de-escalate
            result = await session.run(user_input="thanks")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert switchboard.current_model == "fast"

    async def test_switch_events_fire(self, fast_llm, smart_llm):
        decision_events: list[SwitchEvent] = []
        switch_events: list[SwitchEvent] = []
        sb = Switchboard(
            models={"fast": fast_llm, "smart": smart_llm},
            config=SwitchboardConfig(
                model_topics={"smart": ["pricing"]},
                escalation_model="smart",
                escalation_threshold=0.5,
                on_decision=decision_events.append,
                on_switch=switch_events.append,
            ),
        )

        async with AgentSession() as session:
            await session.start(Agent(instructions=INSTRUCTIONS, llm=sb))

            # Simple message — stays fast
            result = await session.run(user_input="Hello")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert len(decision_events) == 1
            assert decision_events[0].to_model == "fast"
            assert decision_events[0].changed is False
            assert len(switch_events) == 0  # no change, on_switch not fired

            # Pricing — escalates to smart
            result = await session.run(user_input="What about pricing?")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert len(decision_events) == 2
            assert decision_events[1].to_model == "smart"
            assert decision_events[1].changed is True
            assert len(switch_events) == 1  # change! on_switch fired
