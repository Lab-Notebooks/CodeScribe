# Model backends

This page documents the current backend selection logic in
`codescribe/lib/_llm.py`.

## Supported model inputs

`set_neural_model(model, reasoning=False)` currently accepts:

- `openai-*`
- `oaic-*`
- `anthropic-*`

If none of these match, CodeScribe raises `ValueError`.

## Common output-token setting

All backends read `CODESCRIBE_MAX_TOKENS`. Both backends currently default to
the same value:

- `OpenAICompModel`: default `32768`, sent as the Responses API's
  `max_output_tokens` (covers visible output plus reasoning tokens)
- `AnthropicModel`: default `32768`, sent as `max_tokens`

## `openai-*`

Backend class:

- `OpenAICompModel(..., profile="openai")`

Required environment variable:

- `OPENAI_API_KEY`

Behavior:

- uses the OpenAI **Responses API** (`client.responses.create(...)`), not
  Chat Completions,
- supports tool use through the unified `chat_with_tools(...)` path (function
  tools are sent in the flat Responses shape),
- system messages are pulled out of the chat template into the top-level
  `instructions` field,
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

- targets an OpenAI-compatible API that implements `/v1/responses`,
- uses the same Responses API code path as hosted OpenAI with a different
  profile/base URL,
- supports tool use through the same `chat_with_tools(...)` interface.

### Reasoning support (`openai-*` / `oaic-*`)

`--reason` on `agent` and `loop` is passed into
`set_neural_model(..., reasoning=True)`, which fixes `reasoning_effort="high"`
and `reasoning_summary="auto"`. There is no way to configure a different
effort or summary level.

When reasoning is enabled, requests set `reasoning={"effort", "summary"}` and
`include=["reasoning.encrypted_content"]`. Returned reasoning items are kept
verbatim (encrypted content is opaque) and replayed on the next turn via
`format_tool_result_messages(...)`, the same pattern `AnthropicModel` uses for
`thinking` blocks.

### Prompt caching (`openai-*` / `oaic-*`)

Always on, unconditionally — there is no environment variable toggle and no
separate "explicit mode" flag. Every request sets `prompt_cache_options:
{"mode": "explicit", "ttl": "30m"}` and stamps a `prompt_cache_breakpoint` on
the same second-to-last user message `AnthropicModel` marks with
`cache_control`, matching cache behavior between the two backends as closely
as the two APIs allow. Explicit breakpoints are supported on `gpt-5.6`+
models.

Requests always send `store: false` — CodeScribe resends the full
conversation itself every turn (it does not use `previous_response_id`), so
there is nothing for OpenAI to retain server-side, matching how
`AnthropicModel` has no server-side storage concept at all.

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
- prompt caching is always on (no environment variable toggle),
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

- OpenAI-compatible backends use the Responses API's flat function-tool shape
  (`function_call`/`function_call_output` items), not Chat Completions' nested
  `tool_calls` shape.
- Anthropic backends use Anthropic tool APIs.
