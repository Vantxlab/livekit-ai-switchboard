# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (with dev dependencies)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_analyzer.py

# Run a single test class or method
pytest tests/test_switchboard.py::TestBasicRouting::test_short_simple_message_stays_on_fast -v
```

There is no build step, linter, or formatter configured.

## Architecture

This is an intelligent LLM routing library for LiveKit voice agents. The `Switchboard` class is a drop-in `llm.LLM` subclass that wraps two LLM instances (fast and smart) and routes each `chat()` call based on heuristic signal analysis and custom rules.

**Routing flow per turn (in `switchboard.py`):**
1. Extract last user message from `ChatContext`
2. Build a `Context` snapshot (immutable dataclass in `context.py`)
3. `HeuristicAnalyzer.analyze()` detects signals and computes a weighted score (0.0–1.0)
4. Custom `Rule` objects are evaluated in descending priority order — first match wins and overrides heuristics
5. Threshold logic: score ≥ escalation_threshold → smart, score ≤ deescalation_threshold → fast, otherwise hold
6. Cooldown guard prevents de-escalation for N turns after escalation
7. `SwitchEvent` emitted via callback for observability
8. `chat()` forwarded to the chosen LLM

**Key modules in `ai_switchboard/`:**
- `switchboard.py` — Core router, state management, LLM delegation
- `analyzer.py` — Detects 8 signal types via phrase matching, computes weighted score
- `signal.py` — Signal name constants, weight table, and phrase lookup dictionaries
- `context.py` — Per-turn conversation snapshot dataclass passed to rules
- `config.py` — `SwitchboardConfig` dataclass (thresholds, cooldown, topics, callbacks)
- `rule.py` — `Rule` dataclass with callable condition, target model, and priority
- `events.py` — `SwitchEvent` dataclass for observability

**State across turns** (`Switchboard` instance): `_current_model`, `_turn_count`, `_turns_on_current`, `_last_switch_turn`, `_interruption_count`, `_repeat_request_count`. Interruption count resets each turn; repeat count accumulates.

## Dependencies

- Single runtime dependency: `livekit-agents>=1.0`
- Python ≥3.9
- Tests use `pytest` + `pytest-asyncio` (asyncio_mode = "auto" — all async tests run automatically)

## Testing Patterns

Tests use lightweight `MagicMock` helpers (`_make_mock_llm`, `_make_chat_ctx`) instead of real LLM instances. The `_chat()` helper in `test_switchboard.py` is a convenience wrapper for routing calls. No external services or fixtures are needed to run tests.
