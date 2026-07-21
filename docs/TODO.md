# TODO

Open follow-ups that still appear relevant to the current code.

## Agent/runtime

- Add tests for `Agent.run()` stop conditions:
  - final text
  - max iterations
  - tool budget exhaustion
  - repeated-call blocking
  - invalid tool-argument JSON recovery
- Add direct tests for `RunResult`, `ToolResult`, and `RejectedCall` so future
  refactors do not silently change loop semantics.
- Consider stronger context-compaction beyond the current workspace summary plus
  output truncation.

## Loop

- Add tests for `PromptLoopRunner` early-exit behavior:
  - execution `STATUS: COMPLETE`
  - empty review `pending` and blocker
- Decide whether review should always run after execution, or whether the
  current “skip review on STATUS: COMPLETE” behavior is final API.
- Document or implement a clearer crash-resume story for `.codescribe/loop/`
  artifacts.

## Tools

- Tighten bounded `bash` safety. It still uses `shell=True` after validation.
- Add tests for path-bounding edge cases in `ReadTool`, `GlobTool`, `EditTool`,
  and `WriteTool`.
- Consider whether `EditTool` should explicitly reject overlapping/nested edits
  instead of relying mainly on exact-match uniqueness.

## Models

- Add backend smoke tests for:
  - `OpenAICompModel`
  - `AnthropicModel`
- Document the practical support matrix for reasoning/token accounting across
  providers.

## Docs

- Keep docs focused on code-backed behavior and remove speculative framework
  comparisons unless they are needed and maintained.
- Reconcile `README.rst` examples and defaults with the current CLI and loop
  implementation.
