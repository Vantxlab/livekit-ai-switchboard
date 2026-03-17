<p align="center">
  <h1 align="center">ai-switchboard</h1>
  <p align="center">
    <strong>Intelligent LLM routing for LiveKit voice agents.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/livekit-ai-switchboard/"><img src="https://img.shields.io/pypi/v/livekit-ai-switchboard?color=blue" alt="PyPI"></a>
    <a href="https://pypi.org/project/livekit-ai-switchboard/"><img src="https://img.shields.io/pypi/pyversions/livekit-ai-switchboard" alt="Python"></a>
    <a href="https://github.com/vantxlab/livekit-ai-switchboard/blob/main/LICENSE"><img src="https://img.shields.io/github/license/vantxlab/livekit-ai-switchboard" alt="License"></a>
  </p>
</p>

---

Route conversations between **N named models** based on topics, heuristic signals, and custom rules. Simple turns stay on the fast model for low latency; complex, friction-heavy, or topic-sensitive turns escalate to higher-tier models automatically.

The `Switchboard` is a drop-in [`llm.LLM`](https://docs.livekit.io/agents/) replacement — no extra wiring needed.

## Installation

```bash
pip install livekit-ai-switchboard
```

> **Requires** Python 3.10+ and `livekit-agents >= 1.0`

## Quick Start

```python
from ai_switchboard import Switchboard, SwitchboardConfig
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import groq, openai

switchboard = Switchboard(
    models={
        "fast": groq.LLM(model="llama-3.3-70b-versatile"),
        "smart": openai.LLM(model="gpt-4o"),
    },
    config=SwitchboardConfig(
        model_topics={"smart": ["pricing", "billing"]},
    ),
)

agent = VoicePipelineAgent(llm=switchboard, ...)
```

That's it — `"pricing"` and `"billing"` messages route to the smart model, everything else stays fast.

## Three Layers of Routing

Use as few or as many as you need. Each layer is additive.

### 1. Topics — config only, zero code

Map keywords to models. Any message containing a keyword routes to that model.

```python
Switchboard(
    models={"fast": fast_llm, "standard": std_llm, "premium": premium_llm},
    config=SwitchboardConfig(
        model_topics={
            "premium": ["pricing", "billing", "legal"],
            "standard": ["complaint", "support"],
        },
    ),
)
```

### 2. Heuristic Escalation — config only

Auto-escalate when the conversation gets complex, frustrated, or heated. The Switchboard scores each turn using built-in signal detectors and escalates when the score exceeds your threshold.

```python
SwitchboardConfig(
    escalation_model="premium",
    escalation_threshold=0.6,
)
```

### 3. Rules — opt-in, full control

Lambda conditions for custom logic — VIP routing, time-of-day, metadata checks, etc. Rules take highest priority and bypass cooldown.

```python
from ai_switchboard import Rule

rules = [
    Rule(
        name="vip_customer",
        condition=lambda ctx: "vip" in ctx.last_message.lower(),
        use="premium",
        priority=10,
    ),
]
```

## How Routing Works

Every turn, the Switchboard evaluates in this order:

```
Rules (highest priority first, first match wins)
  ↓ no match
Topic keywords + Heuristic score → pick the higher-tier model
  ↓ nothing triggered
Default model (first in your models dict)
  ↓
Cooldown guard (holds current model for N turns after a switch)
  ↓
Forward chat() to the chosen model
```

Model tier is determined by insertion order — later in the dict means higher tier.

## Heuristic Signals

| Signal | Category | Trigger | Weight |
|---|---|---|---:|
| `long_input` | Complexity | Word count > 25 | 0.20 |
| `multi_question` | Complexity | 2+ question marks | 0.30 |
| `complexity_words` | Complexity | "explain", "compare", "why"... | 0.20 |
| `pushback` | Friction | "no", "that's wrong", "incorrect"... | 0.40 |
| `frustration` | Friction | "confused", "doesn't make sense"... | 0.35 |
| `repeat_request` | Friction | "say that again", "can you repeat"... | 0.25 |
| `interruption` | Conversation | User cut agent off | 0.20 |
| `low_stt_confidence` | Voice | STT confidence below threshold | 0.30 |
| `long_audio_turn` | Voice | Audio duration above threshold | 0.15 |
| `topic_match` | Topic | Developer-defined keyword hit | 0.50 |

Weights are summed and capped at 1.0. Default escalation threshold: **0.60**.

## Full Configuration

```python
from ai_switchboard import Switchboard, SwitchboardConfig

sb = Switchboard(
    models={
        "fast": fast_llm,
        "standard": std_llm,
        "premium": premium_llm,
    },
    config=SwitchboardConfig(
        default_model="fast",               # fallback (default: first in dict)
        cooldown_turns=2,                    # hold after switch before de-escalating

        # Topic routing
        model_topics={
            "premium": ["pricing", "billing"],
            "standard": ["complaint", "support"],
        },

        # Heuristic escalation
        escalation_model="premium",          # escalate to this when score is high
        escalation_threshold=0.6,            # score threshold (0.0–1.0)

        # Voice thresholds
        stt_confidence_threshold=0.7,        # below → low_stt_confidence signal
        long_audio_threshold=10.0,           # seconds above → long_audio_turn signal

        # Observability
        on_switch=my_switch_callback,        # fires only on model change
        on_decision=my_decision_callback,    # fires every turn
        log_decisions=True,                  # log to "ai_switchboard" logger
    ),
)
```

## Voice Signals

The Switchboard detects voice-specific signals from LiveKit's STT pipeline.

**STT Confidence** — low transcription confidence suggests ambiguous input that a smarter model may handle better:

```python
SwitchboardConfig(stt_confidence_threshold=0.7)  # default
```

**Audio Duration** — long audio turns often indicate complex input:

```python
SwitchboardConfig(long_audio_threshold=10.0)  # seconds, default
```

Feed audio duration from your pipeline:

```python
switchboard.record_audio_duration(seconds=12.5)
```

## Observability

Two callbacks for different use cases:

```python
from ai_switchboard import SwitchEvent

def on_switch(event: SwitchEvent):
    """Fires only when the model changes."""
    print(f"Switched {event.from_model} → {event.to_model}")

def on_decision(event: SwitchEvent):
    """Fires every turn, whether or not the model changed."""
    print(f"Turn {event.turn}: model={event.to_model} score={event.heuristic_score:.2f}")
```

Every `SwitchEvent` includes: `from_model`, `to_model`, `turn`, `heuristic_score`, `signals_fired`, `triggered_by`, and `changed`.

## Rule Context

Every rule condition receives a `Context` object:

| Field | Type | Description |
|---|---|---|
| `last_message` | `str` | Raw user message text |
| `last_message_word_count` | `int` | Word count |
| `turn_count` | `int` | Total turns so far |
| `interruption_count` | `int` | Interruptions this turn |
| `repeat_request_count` | `int` | Accumulated repeat requests |
| `current_model` | `str` | Name of current model |
| `turns_on_current_model` | `int` | Turns since last switch |
| `last_switch_turn` | `int` | Turn number of last switch |
| `stt_confidence` | `float \| None` | STT transcript confidence |
| `audio_duration` | `float \| None` | Audio duration in seconds |
| `signals_fired` | `list[str]` | Signals detected this turn |
| `heuristic_score` | `float` | Weighted score (0.0–1.0) |

## API Reference

### `Switchboard`

```python
Switchboard(
    models: dict[str, llm.LLM],     # named models, insertion order = tier order
    config: SwitchboardConfig = ..., # routing configuration
    rules: list[Rule] = [],          # custom routing rules
)
```

| Property / Method | Description |
|---|---|
| `switchboard.current_model` | Name of the active model (`str`) |
| `switchboard.model` | Underlying LLM's model identifier |
| `switchboard.default_model` | Name of the fallback model |
| `switchboard.record_interruption()` | Signal that the user interrupted |
| `switchboard.record_audio_duration(seconds)` | Feed audio turn length |
| `switchboard.reset()` | Clear all state for a new session |

### `Rule`

```python
Rule(
    name: str,                               # identifier for logging
    condition: Callable[[Context], bool],     # evaluated each turn
    use: str,                                # target model name
    priority: int = 5,                       # higher = evaluated first
)
```

## Examples

See the [`examples/`](examples/) directory:

| File | Description |
|---|---|
| `basic.py` | Minimal 2-model setup with topic routing |
| `three_tier.py` | 3-model setup with topics + heuristic escalation |
| `custom_rules.py` | Rules for VIP routing and custom logic |
| `openrouter_anthropic.py` | Groq + Anthropic via OpenRouter |
| `demo_agent_session.py` | LiveKit AgentSession testing pattern |
| `demo_events.py` | Observability with `on_switch` / `on_decision` |

## License

[Apache 2.0](LICENSE)
