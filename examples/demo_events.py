"""Demo — Observability via SwitchEvent callback.

Shows how to capture every routing decision for logging, metrics, or analytics.

Run:  python examples/demo_events.py

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


events_log: list[SwitchEvent] = []


def collect_event(event: SwitchEvent) -> None:
    events_log.append(event)


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
        fast=fast,
        smart=smart,
        config=SwitchboardConfig(
            on_switch=collect_event,
            cooldown_turns=1,
        ),
    )

    messages = [
        "Hi!",
        "What's 2+2?",
        "No, that's wrong. Can you explain why step by step? How does arithmetic work?",
        "ok got it",
        "thanks bye",
    ]

    print("Running conversation...\n")
    for msg in messages:
        reply = await ask(sb, msg)
        print(f"User: {msg}")
        print(f"  -> {sb.current_model}: {reply[:80]}...\n")

    print("=" * 60)
    print("Event log summary:")
    print("=" * 60)
    for e in events_log:
        status = "SWITCHED" if e.changed else "stayed"
        print(
            f"  turn {e.turn}: {e.from_model:>5} -> {e.to_model:<5} "
            f"[{status:<8}] score={e.heuristic_score:.2f} "
            f"signals={e.signals_fired}"
        )

    switches = sum(1 for e in events_log if e.changed)
    print(f"\nTotal turns: {len(events_log)}, model switches: {switches}")


if __name__ == "__main__":
    asyncio.run(main())
