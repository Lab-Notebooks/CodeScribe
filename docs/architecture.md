# Architecture

## Package layout

```text
codescribe/
  cli/           Click commands (_code_scribe.py group, _commands.py subcommands)
  api/           thin public functions; forward into codescribe.lib
  lib/
    agent/       Agent loop, run results, observers, single-session prompt_agent
    loop/        bounded author → review loop: state, guard, check, prompts, runner
    models/      Model base + OpenAI-compatible and Anthropic backends
    tools/       AgentTool / PathTool base + read, glob, bash, edit, write
    util/        logging, telemetry, media attachment, file/prompt helpers
    _cmd.py      prompt-driven commands: translate, inspect, generate, update
```

Each subpackage `__init__.py` lists its public names; `lib/__init__.py`
re-exports them, so callers use `codescribe.lib.<name>`.

Dependencies point one way: `tools`, `models` → `agent` → `loop`, with `util`
used by all.

## Control flow

```text
code-scribe <command> → cli/_commands.py → api/_commands.py → lib
```

- `agent`: `lib/agent/_session.py: prompt_agent()` → `Agent.run()`
- `loop`: `lib/loop/_runner.py: prompt_loop()` → `PromptLoopRunner.run()`
- `translate` / `generate` / `update`: `lib/_cmd.py`, plain `model.chat()`, no tools
- `inspect`: `lib/_cmd.py`, `Agent` with read-only tools

## Class hierarchy

- `Model` → `OpenAICompModel`, `AnthropicModel`. Each normalizes replies to
  `{text, tool_calls, usage, reasoning, reasoning_blocks}`; `Model.api` names the
  content-block format.
- `AgentTool` → `BashTool`, `GuardedBashTool` (loop author only)
- `AgentTool` → `PathTool` (root bounding, protected paths) → `ReadTool`,
  `GlobTool`, `EditTool`, `WriteTool`
- `RunObserver` → `ConsoleObserver`

See [agent.md](agent.md), [loop.md](loop.md), [models.md](models.md) and
[tools.md](tools.md) for details.
