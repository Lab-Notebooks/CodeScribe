# Architecture

This page is a code-grounded map of the current CodeScribe package.

## Package shape

The main layers are:

- `codescribe/cli/`
  - Click command definitions.
  - Top-level command group is in `codescribe/cli/_code_scribe.py`.
  - Subcommands are defined in `codescribe/cli/_commands.py`.
- `codescribe/api/`
  - Thin public API wrappers around library functions.
  - Mostly forwards CLI arguments into `codescribe.lib`.
- `codescribe/lib/`
  - Core implementation.
  - Agent runtime: `_agent.py`
  - Loop runtime: `_loop.py`
  - Tools: `_tools.py`
  - Model backends: `_llm.py`
  - Prompt-driven commands: `_cmd.py`
  - File/prompt helpers: `_filetools.py`, `_textprotocol.py`, `_logging.py`

`codescribe/lib/__init__.py` re-exports these modules so the CLI and API call
into `codescribe.lib.*` symbols directly.

## Control flow

Most commands follow this shape:

```text
code-scribe <command>
  -> codescribe.cli._commands
  -> codescribe.api._commands
  -> codescribe.lib.*
```

Examples:

- `code-scribe agent`
  - `cli/_commands.py: agent()`
  - `api/_commands.py: agent()`
  - `lib/_cmd.py: prompt_agent()`
  - `lib/_agent.py: Agent.run()`
- `code-scribe loop`
  - `cli/_commands.py: loop()`
  - `api/_commands.py: loop()`
  - `lib/_loop.py: prompt_loop()`
- `code-scribe translate`
  - `cli/_commands.py: translate()`
  - `api/_commands.py: translate()`
  - `lib/_cmd.py: prompt_translate()`

## Two workflow families

The codebase currently has two distinct styles of execution.

### 1. Prompt-driven commands

Implemented in `codescribe/lib/_cmd.py`:

- `prompt_translate`
- `prompt_generate`
- `prompt_update`

These construct prompts, call `model.chat(...)`, and parse tagged output.
They do not use the tool-using agent loop.

### 2. Agent-driven commands

Implemented by `codescribe/lib/_agent.py` and `codescribe/lib/_loop.py`:

- `prompt_inspect` uses `Agent` with bounded read-only tools.
- `prompt_agent` uses `Agent` with the default bounded toolset rooted at the
  current working directory.
- `prompt_loop` runs repeated execution/review agent sessions.

## Agent runtime

`codescribe/lib/_agent.py` contains the standalone tool-using runtime.

Core pieces:

- `AgentPolicy`
  - Hard limits and repetition policy.
- `TokenUsage`
  - Normalized token accounting across providers.
- `RunResult`
  - Structured result returned by `Agent.run()`.
- `RunObserver` / `ConsoleObserver`
  - Diagnostics rendering.
- `Agent`
  - Main loop: build messages, call model, execute tool calls, repeat.

Important current behavior:

- The agent requires a model that exposes `chat_with_tools()` and reports
  `supports_native_tools=True`.
- Supported agent backends use provider-native tool calling.
- The agent injects a compact `WORKSPACE CONTEXT` block into the latest user
  message each iteration.
- `Agent.run()` returns a structured `RunResult`, not just a plain string.

## Loop runtime

`codescribe/lib/_loop.py` implements a bounded execution/review loop.

Current design:

- Each execution phase starts a fresh `Agent`.
- Each review phase starts a separate fresh `Agent`.
- Cross-loop state is carried in Python objects inside `PromptLoopRunner`:
  - `loop_summaries`
  - `review_summaries`
  - `pending_items`
- On-disk files under `.codescribe/loop/` are for inspection and resume support,
  not the primary in-memory state relay.

Notable files written by the loop harness:

- `.codescribe/loop/run.toml`
- `.codescribe/loop/state.toml`
- `.codescribe/loop/execution.toml`
- `.codescribe/loop/review_output.toml`
- `.codescribe/loop/review.toml`

## Model layer

`codescribe/lib/_llm.py` selects a backend from the model string.

Supported inputs today:

- `openai-*`
- `oaic-*`
- `anthropic-*`

The common abstraction is:

- `chat(messages)` for prompt-driven commands
- `chat_with_tools(messages, tools)` for agent workflows
- `format_tool_result_messages(...)` for feeding tool outputs back into the
  conversation in provider-specific format

## Tool layer

`codescribe/lib/_tools.py` defines:

- `AgentTool`
- `ReadTool`
- `GlobTool`
- `BashTool`
- `EditTool`
- `WriteTool`

The default constructors are:

- `make_tools(root, bash_allow=None)`
- `make_readonly_tools(root, bash_allow=None)`

Both are bounded to a root path. The difference is that the read-only variant
omits `edit` and `write`.

## Important doc corrections

A few stale descriptions are easy to get wrong when reading older docs:

- `prompt_agent()` currently uses bounded tools rooted at `Path.cwd()`, not an
  unrestricted filesystem toolset.
- `prompt_inspect()` includes bounded `bash` through `make_readonly_tools()`;
  it is not limited to only `read` and `glob`.
- The loop execution phase is instructed to complete as much work as possible in
  one session, not exactly one task per loop.
- The review phase can use `read`, `glob`, `write`, and a tightly restricted
  bounded `bash` tool.
