# Model backends

This page documents the current backend selection logic in
`codescribe/lib/_llm.py`.

## Supported model inputs

`set_neural_model(model, reasoning=False)` currently accepts:

- `openai-*`
- `oaic-*`
- `anthropic-*`

If none of these match, CodeScribe raises `ValueError`.

## At a glance

| | `openai-*` / `oaic-*` | `anthropic-*` |
|---|---|---|
| Class | `OpenAICompModel` | `AnthropicModel` |
| Wire API | Chat Completions (`chat.completions.create`) | Messages (`messages.stream` / `messages.create`) |
| Streaming | attempted first, falls back to non-streaming | attempted first, falls back to non-streaming |
| Output cap | `max_tokens` | `max_tokens` |
| System prompt | inline `system` messages | hoisted to top-level `system` |
| Tool shape | nested `tool_calls` + `role: "tool"` replies | `tool_use` / `tool_result` blocks |
| Reasoning knob | `reasoning_effort="high"` | `thinking={"type": "adaptive", ...}` |
| Prompt caching | provider-automatic, nothing sent | explicit `cache_control` breakpoints |
| Reasoning replay | flattened to plain assistant text | blocks echoed back verbatim |

## Common output-token setting

All backends read `CODESCRIBE_MAX_TOKENS`. Both backends currently default to
the same value:

- `OpenAICompModel`: default `32768`, sent as Chat Completions' `max_tokens`
  (the cap on generated output tokens)
- `AnthropicModel`: default `32768`, sent as `max_tokens`

Both backends read the variable in `__init__`, so the value is resolved per
instance and picks up a change made at any point before the model is
constructed — whether exported in the shell or set programmatically.

`OpenAICompModel.outputs` (sent as `n=1`) remains a class-level constant; it is
not environment-driven.

## `openai-*`

Backend class:

- `OpenAICompModel(..., profile="openai")`

Required environment variable:

- `OPENAI_API_KEY`

Behavior:

- uses the OpenAI **Chat Completions API**
  (`client.chat.completions.create(...)`), not the Responses API,
- always attempts a streaming call first, asking for
  `stream_options={"include_usage": True}` so the stream reports token usage,
  and degrades in two steps — to a stream without that option if the server
  rejects the field, then to a plain (non-streaming) call if streaming fails
  outright or cannot be parsed,
- system messages stay inline in `messages` (there is no top-level
  `instructions` field); every request also sends `n=1`,
- supports tool use through the unified `chat_with_tools(...)` path (function
  tools are sent in the Chat Completions nested `tool_calls` shape),
- normalizes usage including reasoning-token and cache read/write counts when
  the provider exposes them.

## `oaic-*`

Backend class:

- `OpenAICompModel(..., profile="oaic")`

Required environment variables:

- `OPENAI_COMP_BASEURL`
- `OPENAI_COMP_PROVIDER`
- `OPENAI_COMP_APIKEY`

Behavior:

- targets an OpenAI-compatible API that implements `/v1/chat/completions`,
- uses the same Chat Completions code path as hosted OpenAI with a different
  profile/base URL,
- supports tool use through the same `chat_with_tools(...)` interface.

`OPENAI_COMP_PROVIDER` is validated at construction but is not sent on the
wire; it exists so a missing routing label fails fast rather than mid-run.

### The two profiles are the same code path

`profile` is read only in `__init__` and `__repr__`. Nothing in request
building, streaming, tool formatting, or usage normalization branches on it.
The profile decides exactly two things: which environment variables supply
credentials, and whether `base_url` is passed to the client. For the same model
string the request payload is identical.

Any behavioral difference you observe between `openai-*` and `oaic-*` —
including whether a streamed response reports token usage — therefore comes
from the server on the other end, not from CodeScribe.

### Reasoning support (`openai-*` / `oaic-*`)

`--reason` on `agent` and `loop` is passed into
`set_neural_model(..., reasoning=True)`, which fixes `reasoning_effort="high"`.
There is no way to configure a different effort level.

When reasoning is enabled, requests set the top-level `reasoning_effort` field.
No summary or encrypted-reasoning parameters are sent — reasoning text is only
surfaced when the provider volunteers it (a `reasoning` field on the message or
streamed delta, or `reasoning`/`summary_text` content blocks).

Any reasoning text that does come back is replayed on the next turn by
`format_tool_result_messages(...)` as plain assistant message content. Unlike
`AnthropicModel`, which echoes `thinking` blocks back verbatim, there is no
opaque/encrypted reasoning item to preserve on this path.

### Prompt caching (`openai-*` / `oaic-*`)

CodeScribe sends no cache-control parameters on this path: there is no
`prompt_cache_options`, no `prompt_cache_breakpoint`, and no `store` flag.
Whatever caching happens is the provider's own automatic (implicit) prompt
caching, which matches the longest reusable prefix of the request and places
its own breakpoint at the end of the latest eligible message.

This is a deliberate change from the short-lived Responses API implementation,
which set `prompt_cache_options: {"mode": "explicit", "ttl": "30m"}` and
stamped a single breakpoint on the second-to-last user message. In explicit
mode the provider caches *only* up to the markers you place, so everything
after that one breakpoint — the newest turn and its tool results — was billed
uncached on every request, and explicit mode is only honored on `gpt-5.6`+
models in the first place. Automatic caching generally yields higher hit rates
for an agent loop, whose prefix grows monotonically.

Streaming has no bearing on any of this. Cache lookup is a server-side match on
the request payload prefix; whether the reply is streamed back token-by-token
or returned whole does not change what is cached or reused.

## `anthropic-*`

Backend class:

- `AnthropicModel`

Required environment variable:

- `ANTHROPIC_API_KEY`

Optional environment variables:

- `ANTHROPIC_BASE_URL`
- `CODESCRIBE_MODEL_REASONING`

Current behavior:

- always attempts streaming API calls, falling back to a plain (non-streaming)
  call if the SDK/provider doesn't support it,
- prompt caching is always on (no environment variable toggle): the system
  prompt and the last tool definition are written at a 1h TTL, plus a rolling
  breakpoint on the second-to-last user turn at the default 5m TTL,
- supports tool use through Anthropic tool APIs,
- can relay Anthropic thinking blocks back into the next turn.

### Reasoning support

`--reason` on `agent` and `loop` is passed into `set_neural_model(..., reasoning=True)`.

For `AnthropicModel`, reasoning can be enabled by either:

- the CLI/API `reason=True`, or
- `CODESCRIBE_MODEL_REASONING=1`

When enabled, the model is configured with:

```python
{"type": "adaptive", "display": "summarized"}
```

Returned thinking blocks are preserved and echoed back in
`format_tool_result_messages(...)` as required by the Anthropic API.

## Tool-calling note

All supported agent backends expose the same `chat_with_tools(...)` interface to
`Agent`.

- OpenAI-compatible backends use Chat Completions' nested `tool_calls` shape:
  the assistant message carries `tool_calls` entries
  (`{"id", "type": "function", "function": {"name", "arguments"}}`) and each
  result comes back as a `{"role": "tool", "tool_call_id", "content"}` message.
- Anthropic backends use Anthropic tool APIs (`tool_use` / `tool_result`
  content blocks).

Both backends tolerate malformed tool arguments: if `json.loads` on the
argument string fails, the call is still returned with `arguments = {}` plus
`_raw_arguments` and `_raw_arguments_error` keys, so the agent loop can report
the failure instead of crashing.

## Caveats

### Both backends

- **Streaming failure re-runs the request.** Each backend wraps its streaming
  attempt in a bare `except` and falls back to a non-streaming `create(...)`.
  If the stream dies partway through, the fallback issues a *second* billable
  request; the tokens already generated by the aborted stream are paid for but
  discarded.
- **A silent fallback is invisible.** Nothing logs which path served a given
  turn, so a provider that consistently fails streaming will quietly double
  latency and cost.

### `openai-*` / `oaic-*`

- **Usage reporting on a stream depends on the server.** Chat Completions omits
  the usage chunk unless the request sets
  `stream_options={"include_usage": True}`, so CodeScribe asks for it. A server
  that rejects the unknown field gets a second streaming attempt without it,
  and that stream reports no usage — the iteration then contributes zeros to
  the token totals. This is a property of the endpoint, not of the profile:
  the same request goes out for `openai-*` and `oaic-*`. It affects
  *reporting* only; the provider still bills, and caching still works.
- **Interrupted streams lose usage**, since the usage chunk arrives last.
- **A rejected `stream_options` costs an extra round trip** on every request,
  because the retry is not remembered across calls.
- **Reasoning is lossy across turns.** Reasoning is re-sent as ordinary
  assistant text, so providers that require their own reasoning items back
  verbatim will not see them.
- **`reasoning_summary` is computed but unused.** The constructor sets it when
  `reasoning=True`; `_request_kwargs` never sends it. Only `reasoning_effort`
  reaches the wire.

### `anthropic-*`

- **Reasoning tokens are not captured while streaming.** `_merge_stream_usage`
  reads only `input_tokens`, the two cache counters, and the delta
  `output_tokens` from stream events, so `reasoning` stays `0` on the streaming
  path even with thinking enabled. The non-streaming path maps a
  `thinking_tokens` field to `reasoning_tokens` if the SDK exposes one.
- **Cache TTL ordering is load-bearing.** The system prompt and last tool
  definition are written at `ttl: "1h"` while the rolling message breakpoint
  uses the default 5m. Longer TTLs must precede shorter ones in the prefix;
  reordering these writes will break caching.
- **`extended-cache-ttl-2025-04-11` header is vestigial.** The 1h TTL is
  requested per-block via `cache_control.ttl`. The header is retained only for
  older gateways behind `ANTHROPIC_BASE_URL`.

## Token accounting

`TokenUsage` in `codescribe/lib/_agent.py` normalizes both providers into five
counters — `input`, `output`, `reasoning`, `cache_write`, `cache_read` — which
accumulate across iterations with `+`. `TokenUsage.from_raw` accepts either
naming convention, preferring OpenAI's `prompt_tokens`/`completion_tokens` and
falling back to Anthropic's `input_tokens`/`output_tokens`.

### The total

```python
total = input + output + cache_write + cache_read
```

`reasoning` is deliberately **excluded** from the total. Both providers count
reasoning/thinking tokens inside their output-token figure, so adding it again
would double-count. Treat `reasoning` as a breakdown of `output`, not an extra
term — this is why the verbose `usage` line prints `rsn` separately and does
not fold it into `total`.

### Making the two providers comparable

The two APIs disagree on whether cached tokens are inside the input count:

- **Anthropic** reports `input_tokens` as the tokens *after* the last cache
  breakpoint, with `cache_read_input_tokens` and `cache_creation_input_tokens`
  reported alongside. The three are disjoint, so the sum above is already
  correct.
- **OpenAI** reports `prompt_tokens` as the *whole* prompt, cached portion
  included.

To keep one formula valid for both, `_normalize_openai_usage` subtracts the
cached total from the input figure:

```python
cached_total = cached_tokens + cache_write_tokens
prompt_tokens = max(0, prompt_tokens - cached_total)
```

After this, `input` means "uncached input tokens" on both backends and `total`
is the true billable prompt-plus-completion size.

### Where the numbers are read from

| Counter | OpenAI-compatible | Anthropic |
|---|---|---|
| `input` | `prompt_tokens` / `input_tokens`, minus cached total | `input_tokens` |
| `output` | `completion_tokens` / `output_tokens` | `output_tokens` (from `message_delta` when streaming) |
| `reasoning` | `completion_tokens_details.reasoning_tokens`, or top-level `reasoning_tokens` | `thinking_tokens`, non-streaming only |
| `cache_read` | `prompt_tokens_details.cached_tokens` (or `input_tokens_details`, or top level) | `cache_read_input_tokens` |
| `cache_write` | `cache_write_tokens` / `cache_creation_tokens` (or `cache_creation_input_tokens`) | `cache_creation_input_tokens` |

### Accounting caveats

- **Missing usage reads as zero, not as unknown.** `TokenUsage.from_raw(None)`
  returns an all-zero record, so a turn whose usage never arrived (a stream
  from a server that does not report it) is indistinguishable in the totals
  from a turn that genuinely used nothing. Run totals are a floor, not an exact
  figure.
- **The cache subtraction assumes cached tokens were included.** If an
  OpenAI-compatible provider reports a cache-write counter that is *not* part
  of `prompt_tokens`, the subtraction under-reports `input`. The `max(0, ...)`
  clamp prevents negatives but also hides the discrepancy.
- **Only the first matching cache field is used.** `_attr` returns the first
  non-`None` name it finds, so a provider populating both a nested and a
  top-level counter contributes each value once, not twice.
- **Fallback requests are counted once.** `last_usage` is overwritten by the
  non-streaming retry, so tokens burned by an aborted stream never appear in
  the totals even though they are billed.
- **Cross-provider totals are not cost.** Equal token counts across backends do
  not imply equal spend; cache reads and writes are priced differently from
  fresh input on both providers.
