"""Demo — Switchboard inside a LiveKit AgentSession.

Uses the official testing pattern from docs.livekit.io/agents/start/testing/:
  - async with AgentSession() as session
  - LLM on the Agent, not the session
  - await session.run(user_input=...)
  - result.expect assertions

Run:  python examples/demo_agent_session.py

Requires OPENAI_API_KEY in .env or environment.
"""

from __future__ import annotations

import asyncio
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from livekit.agents.voice import Agent, AgentSession
from livekit.plugins.openai import LLM

from ai_switchboard import Switchboard, SwitchboardConfig, SwitchEvent


def on_switch(event: SwitchEvent) -> None:
    marker = "SWITCHED" if event.changed else "stayed"
    print(
        f"  [{marker}] model={event.to_model} "
        f"score={event.heuristic_score:.2f} signals={event.signals_fired}"
    )


async def main() -> None:
    fast = LLM(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    smart = LLM(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

    sb = Switchboard(
        fast=fast,
        smart=smart,
        config=SwitchboardConfig(cooldown_turns=1, on_switch=on_switch),
    )

    conversation = [
        "Hi, what can you help me with?",
        "No that's wrong. Can you explain why? How does this compare to alternatives?",
        "ok",
        "thanks bye",
    ]

    async with AgentSession() as session:
        agent = Agent(
            instructions="You are a helpful assistant. Keep responses to one sentence.",
            llm=sb,
        )
        await session.start(agent)

        for msg in conversation:
            print(f"\n{'=' * 50}")
            print(f"User: {msg}")
            print(f"{'=' * 50}")

            result = await session.run(user_input=msg)

            # Use LiveKit's assertion framework
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()

            # Print the assistant's response
            response = result.events[0].item.text_content
            print(f"Assistant: {response}")
            print(f"  (current model: {sb.current_model})")

    print("\nSession closed.")


if __name__ == "__main__":
    asyncio.run(main())
