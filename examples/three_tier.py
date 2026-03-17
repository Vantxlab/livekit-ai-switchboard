"""Level 2 — Three-tier routing with topics and heuristic escalation.

Three models (fast / standard / premium) with declarative topic routing
and automatic heuristic escalation for frustration and complexity.
"""

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, groq, openai, silero

from ai_switchboard import Switchboard, SwitchboardConfig, SwitchEvent


def log_switch(event: SwitchEvent) -> None:
    print(
        f"[switchboard] turn {event.turn}: "
        f"{event.from_model} -> {event.to_model} "
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
            models={
                "fast": groq.LLM(model="llama-3.3-70b-versatile"),
                "standard": openai.LLM(model="gpt-4o-mini"),
                "premium": openai.LLM(model="gpt-4o"),
            },
            config=SwitchboardConfig(
                default_model="fast",
                model_topics={
                    "premium": ["pricing", "billing", "legal"],
                    "standard": ["complaint", "support"],
                },
                escalation_model="premium",
                escalation_threshold=0.6,
                cooldown_turns=2,
                on_switch=log_switch,
            ),
        ),
        tts=openai.TTS(),
    )

    agent.start(ctx.room)
    await agent.say("Hey, how can I help you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
