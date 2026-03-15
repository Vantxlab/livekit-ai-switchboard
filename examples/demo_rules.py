"""Demo — Custom rules override heuristics.

Run:  python examples/demo_rules.py

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

from ai_switchboard import Rule, Switchboard, SwitchboardConfig, SwitchEvent


def on_switch(event: SwitchEvent) -> None:
    marker = ">>>" if event.changed else "   "
    print(
        f"  {marker} model={event.to_model:<5} "
        f"triggered_by={event.triggered_by} "
        f"score={event.heuristic_score:.2f}"
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

    rules = [
        Rule(
            name="vip_customer",
            condition=lambda ctx: "vip" in ctx.last_message.lower(),
            use="smart",
            priority=10,
        ),
        Rule(
            name="short_confirm",
            condition=lambda ctx: ctx.last_message_word_count <= 2,
            use="fast",
            priority=5,
        ),
    ]

    sb = Switchboard(
        fast=fast,
        smart=smart,
        config=SwitchboardConfig(on_switch=on_switch, cooldown_turns=0),
        rules=rules,
    )

    conversation = [
        "Hello there!",
        "I'm a VIP customer, I need help with my account.",
        "yes",
        "Can you explain the difference between your enterprise and starter plans?",
        "ok thanks",
    ]

    for msg in conversation:
        print(f"\nUser: {msg}")
        reply = await ask(sb, msg)
        print(f"Assistant ({sb.current_model}): {reply[:120]}...")


if __name__ == "__main__":
    asyncio.run(main())
