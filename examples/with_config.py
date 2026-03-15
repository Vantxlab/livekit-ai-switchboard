"""Level 2 — Tune routing behavior.

Customize thresholds, cooldown, topic-based escalation, and add
observability via the ``on_switch`` callback.
"""

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, groq, openai, silero

from ai_switchboard import Switchboard, SwitchboardConfig, SwitchEvent


def log_switch(event: SwitchEvent) -> None:
    if event.changed:
        print(
            f"[switchboard] turn {event.turn}: "
            f"{event.from_model} → {event.to_model} "
            f"(score={event.heuristic_score:.2f}, "
            f"signals={event.signals_fired}, "
            f"triggered_by={event.triggered_by})"
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=Switchboard(
            fast=groq.LLM(model="llama-3.3-70b-versatile"),
            smart=openai.LLM(model="gpt-4o"),
            config=SwitchboardConfig(
                escalation_threshold=0.5,
                cooldown_turns=3,
                smart_topics=["pricing", "warranty", "complaint"],
                on_switch=log_switch,
            ),
        ),
        tts=openai.TTS(),
    )

    agent.start(ctx.room)
    await agent.say("Hey, how can I help you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
