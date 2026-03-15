"""Integration tests using LiveKit's AgentSession testing framework.

Uses the official pattern from docs.livekit.io/agents/start/testing/:
  - async with AgentSession() as session
  - LLM on the Agent, not the session
  - await session.run(user_input=...)
  - result.expect assertions

Requires OPENAI_API_KEY in the environment (or .env file).
Run with: pytest tests/test_integration.py -m integration -v
"""

from __future__ import annotations

import pytest
from livekit.agents.voice import Agent, AgentSession

from ai_switchboard import Rule, Switchboard, SwitchboardConfig, SwitchEvent
from tests.conftest import skip_no_openai

pytestmark = [pytest.mark.integration, skip_no_openai]

INSTRUCTIONS = "You are a helpful assistant. Keep responses very brief."


class TestAgentSessionRouting:
    async def test_simple_chat_stays_fast(self, fast_llm, smart_llm):
        sb = Switchboard(fast=fast_llm, smart=smart_llm)

        async with AgentSession() as session:
            await session.start(Agent(instructions=INSTRUCTIONS, llm=sb))

            result = await session.run(user_input="Hi there!")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert sb.current_model == "fast"

    async def test_complex_message_escalates_to_smart(self, fast_llm, smart_llm):
        sb = Switchboard(fast=fast_llm, smart=smart_llm)

        async with AgentSession() as session:
            await session.start(Agent(instructions=INSTRUCTIONS, llm=sb))

            result = await session.run(
                user_input=(
                    "No, that's wrong. Can you explain why this happens? "
                    "And how does it compare to the alternative?"
                )
            )
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert sb.current_model == "smart"

    async def test_escalation_then_deescalation(self, fast_llm, smart_llm):
        sb = Switchboard(
            fast=fast_llm,
            smart=smart_llm,
            config=SwitchboardConfig(cooldown_turns=1),
        )

        async with AgentSession() as session:
            await session.start(Agent(instructions=INSTRUCTIONS, llm=sb))

            # Escalate
            result = await session.run(user_input="No, that's wrong. Why? How?")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert sb.current_model == "smart"

            # Cooldown turn
            result = await session.run(user_input="ok")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()

            # De-escalate
            result = await session.run(user_input="thanks")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert sb.current_model == "fast"

    async def test_custom_rule_forces_smart(self, fast_llm, smart_llm):
        rule = Rule(
            name="billing",
            condition=lambda c: "billing" in c.last_message.lower(),
            use="smart",
            priority=10,
        )
        sb = Switchboard(fast=fast_llm, smart=smart_llm, rules=[rule])

        async with AgentSession() as session:
            await session.start(Agent(instructions=INSTRUCTIONS, llm=sb))

            result = await session.run(user_input="I have a billing question")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert sb.current_model == "smart"

    async def test_switch_event_callback_fires(self, fast_llm, smart_llm):
        events: list[SwitchEvent] = []
        sb = Switchboard(
            fast=fast_llm,
            smart=smart_llm,
            config=SwitchboardConfig(on_switch=events.append),
        )

        async with AgentSession() as session:
            await session.start(Agent(instructions=INSTRUCTIONS, llm=sb))

            # Simple message — stays fast
            result = await session.run(user_input="Hello")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert len(events) == 1
            assert events[0].to_model == "fast"
            assert events[0].changed is False

            # Complex message — escalates
            result = await session.run(
                user_input="No that's wrong. Can you explain why? How does this compare?"
            )
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()
            assert len(events) == 2
            assert events[1].to_model == "smart"
            assert events[1].changed is True
