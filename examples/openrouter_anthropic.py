"""OpenRouter Groq (fast) + Anthropic Sonnet 4.6 (smart).

Routes to the smart model when the user says "pricing" or interrupts.
Simple, real-world setup for a sales/support voice agent.

Requires:
  - OPENROUTER_API_KEY in .env or environment
"""

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero

from ai_switchboard import Rule, Switchboard, SwitchboardConfig, SwitchEvent


def on_switch(event: SwitchEvent) -> None:
    print(
        f"[switchboard] {event.from_model} -> {event.to_model} "
        f"(triggered_by={event.triggered_by})"
    )


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Fast model — Groq via OpenRouter (cheap, low-latency)
    fast = openai.LLM.with_openrouter(model="groq/llama-3.3-70b-versatile")

    # Smart model — Anthropic Sonnet 4.6 via OpenRouter
    smart = openai.LLM.with_openrouter(model="anthropic/claude-sonnet-4-6")

    sb = Switchboard(
        models={"fast": fast, "smart": smart},
        config=SwitchboardConfig(
            model_topics={"smart": ["pricing"]},
            escalation_model="smart",
            escalation_threshold=0.5,
            cooldown_turns=2,
            on_switch=on_switch,
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

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=sb,
        tts=openai.TTS(),
    )

    agent.start(ctx.room)
    await agent.say("Hey, how can I help you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
