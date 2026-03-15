"""Level 3 — Custom routing rules.

Define explicit rules that override heuristic scoring. Rules are
evaluated in priority order; the first match wins.
"""

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, groq, openai, silero

from ai_switchboard import Context, Rule, Switchboard


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=Switchboard(
            fast=groq.LLM(model="llama-3.3-70b-versatile"),
            smart=openai.LLM(model="gpt-4o"),
            rules=[
                Rule(
                    name="objection_handling",
                    condition=lambda ctx: any(
                        w in ctx.last_message.lower()
                        for w in ["too expensive", "not interested"]
                    ),
                    use="smart",
                    priority=10,
                ),
                Rule(
                    name="simple_confirmation",
                    condition=lambda ctx: (
                        ctx.last_message_word_count < 5
                        and ctx.current_model == "smart"
                    ),
                    use="fast",
                    priority=5,
                ),
            ],
        ),
        tts=openai.TTS(),
    )

    agent.start(ctx.room)
    await agent.say("Hey, how can I help you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
