# Agent internals

This document describes the current agent implementation in
`codescribe/lib/_agent.py`.

## What the agent does

The agent is an iterative tool-using runtime:

1. Build a message list from system prompt, optional chat history, and task.
2. Call `model.chat_with_tools(messages, tool_schemas)`.
3. If the model requests tools, execute them.
4. Feed tool results back into the conversation.
5. Repeat until final text or a stop condition.

The main entry point is `Agent.run(...)`.

## Core data structures

### `AgentPolicy`

Execution policy knobs:

- `max_tool_calls_total` default `120`
- `max_calls_per_iteration` default `10`
- `max_repeated_calls` default `2`
- `read_repeat_multiplier` default `3`
- `max_consecutive_error_iters` default `3`
- `max_history_chars` default `8000`

### `TokenUsage`

A normalized token-usage record with:

- `input`
- `output`
- `reasoning`
- `cache_write`
- `cache_read`

`TokenUsage.from_raw(...)` normalizes provider usage dictionaries into this
shape.

### `RunResult`

`Agent.run()` returns:

- `final_text`
- `stop_reason`
- `usage`
- `iterations`
- `tool_results`
- `rejected_calls`

`stop_reason` is one of:

- `final_text`
- `max_iterations`
- `tool_budget`

### `ToolResult` and `RejectedCall`

These are intentionally separate.

- `ToolResult` records tool calls that actually executed.
- `RejectedCall` records calls the harness refused before execution.

Current rejection reasons in the code:

- `repeat_blocked`
- `bad_json`
- `iteration_skip`

## Tool-calling contract

The agent requires a model object that:

- is an instance of one of `lib.ALLOWED_MODEL_TYPES`, and
- exposes `chat_with_tools(...)`, and
- reports `supports_native_tools=True`.

Important clarification:

- In this codebase, `supports_native_tools=True` means the backend can
  participate in the agent tool loop through the unified interface.
- For the supported OpenAI-compatible and Anthropic backends, this is
  provider-native tool calling.

## System prompt and workspace context

The agent always appends a built-in `_REACT_NUDGE` rules block to the provided
system prompt.

Each iteration it also injects a compact `WORKSPACE CONTEXT` block into the most
recent user message. That block includes:

- current iteration number,
- total tool calls used,
- recent tool results,
- recent errors.

This is done by `upsert_workspace_context(...)`.

## Execution flow inside `Agent.run()`

Per iteration:

1. Emit `iteration_start` to the log sink.
2. Refresh the `WORKSPACE CONTEXT` block.
3. Call `model.chat_with_tools(...)`.
4. Normalize usage information.
5. If tool calls are present:
   - execute up to `max_calls_per_iteration`,
   - append provider-specific tool result messages,
   - continue to the next iteration.
6. If there are no tool calls and non-empty text exists, stop with `final_text`.
7. If neither tool calls nor text are returned, inject a correction nudge once
   and continue.

## Tool execution and validation

`execute_tool(...)` performs several checks:

- unknown tool rejection,
- disabled tool rejection,
- argument type validation,
- schema validation via `validate_schema_value(...)`.

It emits:

- `tool_start`
- `tool_end`

to the configured logging sink.

A tool return string beginning with `Error:` is treated as a failed tool call
even if the tool implementation did not raise an exception.

## Repetition and budgets

The agent tracks repeated identical `(tool, args)` pairs.

Current policy:

- most tools are blocked after `2` identical calls,
- `read` gets `2 * 3 = 6` tries,
- successful `edit` or `write` resets repetition counts because the workspace
  changed.

Other limits:

- total executed tool calls are capped at `120`,
- per-iteration tool executions are capped at `10`.

Calls beyond the per-iteration cap are recorded as rejected with reason
`iteration_skip`.

## Tool output handling

Tool outputs are recorded in two different ways.

### Full execution channel

Tool outputs are passed back to the model in provider-specific tool-result
messages.

### Compact context channel

`summarize_tool_output(...)` creates short summaries only for the injected
`WORKSPACE CONTEXT` block.

Large successful outputs are truncated before reinsertion into message history if
longer than `max_history_chars`.

Errors are passed through in full.

## Stuck-loop detection

If every real tool call fails for `max_consecutive_error_iters` consecutive
iterations, the agent injects a nudge telling the model to stop calling tools
and emit a `BLOCKED:` section.

This is a behavioral nudge, not a separate stop condition enforced by the
harness.

## Diagnostics and observers

Verbose output is handled by `ConsoleObserver`.

It shows:

- per-iteration progress,
- token usage,
- model reasoning text when present,
- tool starts and one-line result hints,
- final run token totals.

Structured logging is separate from console rendering and is routed through the
configured log sink.

## Runtime tool control

`Agent` exposes:

- `enable_tool(name)`
- `disable_tool(name)`
- `enabled_tools()`
- `tool_schemas()`

These control which tools are advertised and executable.

## Important current limitations

The source currently supports these behaviors; docs should not claim more:

- The agent is not an OS sandbox.
- Context compaction is limited to the fixed-size `WORKSPACE CONTEXT` summary
  plus history-output truncation.
- The agent does not implement a separate planning model or subagent system.
- The agent runtime is single-process and synchronous.

## Related docs

- [`tools.md`](./tools.md)
- [`loop.md`](./loop.md)
- [`models.md`](./models.md)
