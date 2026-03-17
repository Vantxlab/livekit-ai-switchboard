"""Demo — Watch the switchboard route messages in real time.

Run:  python examples/demo_routing.py

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

from livekit.agents.llm import ChatContext
from livekit.plugins.openai import LLM

from ai_switchboard import Switchboard, SwitchboardConfig, SwitchEvent


def on_decision(event: SwitchEvent) -> None:
    marker = ">>>" if event.changed else "   "
    print(
        f"  {marker} turn={event.turn} model={event.to_model:<8} "
        f"score={event.heuristic_score:.2f} signals={event.signals_fired} "
        f"triggered_by={event.triggered_by}"
    )


async def ask(sb: Switchboard, text: str) -> str:
    ctx = ChatContext()
    ctx.add_message(role="user", content=text)
    stream = sb.chat(chat_ctx=ctx)
    chunks: list[str] = []
    async for tok in stream.to_str_iterable():
        chunks.append(tok)
    return "".join(chunks)


async def main() -> None:
    fast = LLM(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    smart = LLM(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

    sb = Switchboard(
        models={"fast": fast, "smart": smart},
        config=SwitchboardConfig(
            cooldown_turns=1,
            on_decision=on_decision,
            model_topics={"smart": ["pricing", "refund"]},
            escalation_model="smart",
            escalation_threshold=0.6,
        ),
    )

    conversation = [
        "Hi, what's the weather like?",
        "Tell me about your pricing plans and how they compare to competitors.",
        "No that's wrong, can you explain the difference more carefully?",
        "ok",
        "thanks",
    ]

    for msg in conversation:
        print(f"\nUser: {msg}")
        reply = await ask(sb, msg)
        print(f"Assistant ({sb.current_model}): {reply[:120]}...")


if __name__ == "__main__":
    asyncio.run(main())
