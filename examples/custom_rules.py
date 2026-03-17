"""Level 3 — Custom routing rules.

Rules are the opt-in escape hatch for logic that can't be expressed
as topics or heuristic thresholds. Evaluated in priority order; first match wins.
Rules bypass cooldown.
"""

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, groq, openai, silero

from ai_switchboard import Rule, Switchboard, SwitchboardConfig


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
                model_topics={"standard": ["support"]},
                escalation_model="premium",
                escalation_threshold=0.6,
            ),
            rules=[
                Rule(
                    name="objection_handling",
                    condition=lambda ctx: any(
                        w in ctx.last_message.lower()
                        for w in ["too expensive", "not interested"]
                    ),
                    use="premium",
                    priority=10,
                ),
                Rule(
                    name="simple_confirmation",
                    condition=lambda ctx: ctx.last_message_word_count < 3,
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
