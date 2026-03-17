"""Demo — Observability via SwitchEvent callbacks.

Shows how to capture every routing decision for logging, metrics, or analytics.
Demonstrates the difference between on_switch (changes only) and on_decision (every turn).

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


switch_log: list[SwitchEvent] = []
decision_log: list[SwitchEvent] = []


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
            on_switch=switch_log.append,
            on_decision=decision_log.append,
            model_topics={"smart": ["pricing"]},
            cooldown_turns=1,
        ),
    )

    messages = [
        "Hi!",
        "What's 2+2?",
        "Tell me about pricing",
        "ok got it",
        "thanks bye",
    ]

    print("Running conversation...\n")
    for msg in messages:
        reply = await ask(sb, msg)
        print(f"User: {msg}")
        print(f"  -> {sb.current_model}: {reply[:80]}...\n")

    print("=" * 60)
    print("Decision log (every turn):")
    print("=" * 60)
    for e in decision_log:
        status = "SWITCHED" if e.changed else "stayed"
        print(
            f"  turn {e.turn}: {e.from_model:>5} -> {e.to_model:<5} "
            f"[{status:<8}] score={e.heuristic_score:.2f} "
            f"signals={e.signals_fired}"
        )

    print(f"\nTotal turns: {len(decision_log)}, model switches: {len(switch_log)}")


if __name__ == "__main__":
    asyncio.run(main())
