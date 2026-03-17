"""Level 1 — Minimal two-model setup with topic routing.

Drop-in LLM router with sensible defaults. Simple messages go to the fast
model, topic-sensitive turns route to the smart model automatically.
"""

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, groq, openai, silero

from ai_switchboard import Switchboard, SwitchboardConfig


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
                model_topics={"smart": ["pricing", "billing", "warranty"]},
            ),
        ),
        tts=openai.TTS(),
    )

    agent.start(ctx.room)
    await agent.say("Hey, how can I help you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
