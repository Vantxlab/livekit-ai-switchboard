# AI Switchboard Bottleneck Fixes — Design

## Batch 1: Regex fix, configurable weights, min signals

### #1/#6 — Word-boundary matching + precompiled regex

- `HeuristicAnalyzer.__init__()` pre-compiles one `re.Pattern` per signal key using `\b` boundaries for ALL phrases
- Multi-word phrases like `"what if"` become `r"\bwhat\s+if\b"`
- `_match_words()` becomes an instance method using pre-compiled patterns
- Topic matching in analyzer also gets word-boundary treatment

### #4 — Configurable signal weights

- Add `signal_weights: dict[str, float] | None = None` to `SwitchboardConfig`
- Analyzer merges: `{**SIGNAL_WEIGHTS, **(config.signal_weights or {})}`

### #7 — Minimum signals for escalation

- Add `min_signals_for_escalation: int = 1` to `SwitchboardConfig`
- Gate heuristic escalation on `len(ctx.signals_fired) >= config.min_signals_for_escalation`
- Rules and topic routing bypass this gate

## Batch 2: Context enrichment

### #2 — Sliding window of recent messages

- Add `recent_messages: list[str]` to `Context`
- Add `context_window_size: int = 5` to `SwitchboardConfig`
- `Switchboard` maintains `_recent_messages: list[str]` deque, populates Context from it
- `reset()` clears the deque

### #8 — Pass chat context to rules

- Add `chat_ctx: Any | None = None` to `Context`
- `switchboard.py` sets `ctx.chat_ctx = chat_ctx` when building Context
- Rules can inspect full chat history, tool calls, assistant responses

## Batch 3: Runtime intelligence

### #3 — Latency-aware routing

- Add `max_ttfb_ms: dict[str, float] | None = None` to `SwitchboardConfig` (model_name -> max TTFB)
- Add `timeout_fallback_model: str = ""` to `SwitchboardConfig`
- `Switchboard` tracks rolling average TTFB per model in `_model_ttfb: dict[str, list[float]]`
- New method `record_ttfb(model: str, ms: float)` for external reporting
- Before forwarding to chosen model, check if its avg TTFB exceeds threshold — if so, demote to fallback
- Add `ttfb_window_size: int = 10` to config (rolling window length)

### #5 — Built-in metrics collector

- New `metrics.py` module with `SwitchboardMetrics` dataclass
- Tracks: turns per model, switches per trigger type, avg heuristic score, total turns
- `Switchboard` auto-updates metrics on every decision (internal, not via callback)
- Expose via `Switchboard.metrics` property (read-only)
- `reset()` also resets metrics
