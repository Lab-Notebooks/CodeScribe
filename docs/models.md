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

- `OpenAICompModel`: default `32768`
- `AnthropicModel`: default `32768`

## `openai-*`

Backend class:

- `OpenAICompModel(..., profile="openai")`

Required environment variable:

- `OPENAI_API_KEY`

Behavior:

- uses OpenAI chat completions,
- supports tool use through the unified `chat_with_tools(...)` path,
- normalizes usage including reasoning-token counts when the provider exposes
  them.

## `oaic-*`

Backend class:

- `OpenAICompModel(..., profile="oaic")`

Required environment variables:

- `OPENAI_COMP_BASEURL`
- `OPENAI_COMP_PROVIDER`
- `OPENAI_COMP_APIKEY`

Behavior:

- targets an OpenAI-compatible API,
- uses the same client code path as hosted OpenAI with a different profile,
- supports tool use through the same `chat_with_tools(...)` interface.

## `anthropic-*`

Backend class:

- `AnthropicModel`

Required environment variable:

- `ANTHROPIC_API_KEY`

Optional environment variables:

- `ANTHROPIC_BASE_URL`
- `CODESCRIBE_ANTHROPIC_STREAMING`
- `CODESCRIBE_PROMPT_CACHE`
- `CODESCRIBE_MODEL_REASONING`

Current behavior:

- prefers streaming API calls when enabled,
- supports prompt caching,
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

- OpenAI-compatible backends use OpenAI-style tool APIs.
- Anthropic backends use Anthropic tool APIs.
