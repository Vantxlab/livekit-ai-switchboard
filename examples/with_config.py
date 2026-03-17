"""Full configuration — voice thresholds, callbacks, and logging.

Shows all config options including voice signal thresholds,
both callbacks (on_switch + on_decision), and decision logging.
"""

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, groq, openai, silero

from ai_switchboard import Switchboard, SwitchboardConfig, SwitchEvent


def log_switch(event: SwitchEvent) -> None:
    print(
        f"[SWITCH] turn {event.turn}: "
        f"{event.from_model} -> {event.to_model} "
        f"(triggered_by={event.triggered_by})"
    )


def log_decision(event: SwitchEvent) -> None:
    print(
        f"[decision] turn {event.turn}: model={event.to_model} "
        f"score={event.heuristic_score:.2f} signals={event.signals_fired}"
    )


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=Switchboard(
            models={
                "fast": groq.LLM(model="llama-3.3-70b-versatile"),
                "smart": openai.LLM(model="gpt-4o"),
            },
            config=SwitchboardConfig(
                model_topics={"smart": ["pricing", "warranty", "complaint"]},
                escalation_model="smart",
                escalation_threshold=0.5,
                cooldown_turns=3,
                stt_confidence_threshold=0.7,
                long_audio_threshold=10.0,
                on_switch=log_switch,
                on_decision=log_decision,
                log_decisions=True,
            ),
        ),
        tts=openai.TTS(),
    )

    agent.start(ctx.room)
    await agent.say("Hey, how can I help you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
