# ai-switchboard

**Intelligent LLM routing for LiveKit voice agents.**

Route conversations between a fast model (e.g. Groq/Llama) and a smart model (e.g. GPT-4o) based on real-time heuristic signals and custom rules. Simple turns stay on the fast model for low latency; complex, friction-heavy, or topic-sensitive turns escalate to the smart model automatically.

```
pip install livekit-ai-switchboard
```

## Quick Start

```python
from ai_switchboard import Switchboard
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import groq, openai

agent = VoicePipelineAgent(
    llm=Switchboard(
        fast=groq.LLM(model="llama-3.3-70b-versatile"),
        smart=openai.LLM(model="gpt-4o"),
    ),
    ...
)
```

That's it. The Switchboard is a drop-in `llm.LLM` replacement — no extra wiring needed.

## How It Works

Every turn, the Switchboard:

1. Extracts the last user message from the `ChatContext`
2. Runs 8 heuristic signal detectors (complexity, friction, conversation health, topic)
3. Evaluates developer-defined rules in priority order
4. Applies threshold logic and cooldown guards
5. Forwards `chat()` to the chosen model and returns its `LLMStream`

## Heuristic Signals

| Signal | Category | Trigger | Weight |
|--------|----------|---------|--------|
| `long_input` | Complexity | Word count > 25 | 0.20 |
| `multi_question` | Complexity | 2+ question marks | 0.30 |
| `complexity_words` | Complexity | "explain", "compare", "why"… | 0.20 |
| `pushback` | Friction | "no", "that's wrong", "incorrect"… | 0.40 |
| `frustration` | Friction | "confused", "doesn't make sense"… | 0.35 |
| `repeat_request` | Friction | "say that again", "can you repeat"… | 0.25 |
| `interruption` | Conversation | User cut agent off | 0.20 |
| `topic_match` | Topic | Developer-defined keyword hit | 0.50 |

Weights are summed (capped at 1.0). Default escalation threshold: **0.60**.

## Configuration

```python
from ai_switchboard import Switchboard, SwitchboardConfig

Switchboard(
    fast=fast_llm,
    smart=smart_llm,
    config=SwitchboardConfig(
        escalation_threshold=0.5,    # score to switch fast → smart
        deescalation_threshold=0.2,  # score to switch smart → fast
        cooldown_turns=3,            # min turns on smart before de-escalating
        start_on="fast",             # initial model
        smart_topics=["pricing", "warranty", "complaint"],
        on_switch=my_callback,       # called on every routing decision
        log_decisions=True,          # log to "ai_switchboard" logger
    ),
)
```

## Custom Rules

Rules override heuristic scoring. Evaluated in priority order; first match wins.

```python
from ai_switchboard import Switchboard, Rule, Context

Switchboard(
    fast=fast_llm,
    smart=smart_llm,
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
)
```

## Context Object

Every rule condition receives a `Context` with:

| Field | Type | Description |
|-------|------|-------------|
| `last_message` | `str` | Raw user message text |
| `last_message_word_count` | `int` | Word count of last message |
| `turn_count` | `int` | Total turns so far |
| `interruption_count` | `int` | Times user interrupted this turn |
| `repeat_request_count` | `int` | Accumulated repeat requests |
| `current_model` | `str` | `"fast"` or `"smart"` |
| `turns_on_current_model` | `int` | Turns since last switch |
| `last_switch_turn` | `int` | Turn number of last switch (-1 if never) |
| `signals_fired` | `list[str]` | Signals detected this turn |
| `heuristic_score` | `float` | Weighted score (0.0–1.0) |

## Observability

Subscribe to routing decisions via `SwitchEvent`:

```python
from ai_switchboard import SwitchEvent

def on_switch(event: SwitchEvent):
    if event.changed:
        print(f"Switched {event.from_model} → {event.to_model}")
        print(f"  Score: {event.heuristic_score:.2f}")
        print(f"  Signals: {event.signals_fired}")
        print(f"  Triggered by: {event.triggered_by}")
```

## Interruption Tracking

Call `record_interruption()` when a user cuts the agent off mid-speech:

```python
switchboard = Switchboard(fast=fast_llm, smart=smart_llm)
# In your interruption handler:
switchboard.record_interruption()
```

## License

Apache 2.0
