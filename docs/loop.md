# Loop mode

This page documents the current implementation in `codescribe/lib/_loop.py`.

## Overview

`code-scribe loop` runs a bounded author/review workflow over a task file.

The public entry point is:

- `codescribe/lib/_loop.py: prompt_loop()`

The implementation is centered on `PromptLoopRunner`.

## Loop workflow diagram

```
┌─────────────────────────────────────────────────────────┐
│ Loop iteration (1..agent_loops)                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Load task metadata                                  │
│                                                         │
│  2. ┌─────────────────────────────────────────────┐    │
│     │ AUTHOR PHASE                                │    │
│     │ - Fresh Agent with read/glob/bash/edit/write│    │
│     │ - Goal: complete as much work as possible   │    │
│     │ - Emits: STATUS: COMPLETE|INCOMPLETE        │    │
│     │ - Logs: .codescribe/loop/author.toml        │    │
│     └─────────────────────────────────────────────┘    │
│                    │                                    │
│                    ├─→ STATUS: COMPLETE? ──→ STOP      │
│                    │                                    │
│                    └─→ INCOMPLETE                       │
│                         │                               │
│  3. Build LoopSummary from author RunResult            │
│                         │                               │
│  4. ┌─────────────────────────────────────────────┐    │
│     │ REVIEW PHASE                                │    │
│     │ - Fresh Agent with read/glob/write/bash     │    │
│     │ - Verifies author claims                    │    │
│     │ - Writes: review_output.toml                │    │
│     │ - Logs: .codescribe/loop/review.toml        │    │
│     └─────────────────────────────────────────────┘    │
│                    │                                    │
│                    ├─→ No pending + no blocker? ──→ STOP│
│                    │                                    │
│                    └─→ Continue to next loop iteration  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

Cross-loop state carried in-memory: `loop_summaries`, `review_summaries`,
`pending_items`. Injected into next author phase task prompt.

## Current author/review model

For each loop iteration:

1. Reload task metadata if the task file changed.
2. Run an author-phase agent.
3. Build a harness-computed `LoopSummary` from the author `RunResult`.
4. If the author agent reported `STATUS: COMPLETE`, stop early.
5. Otherwise run a review-phase agent.
6. Read `review_output.toml` and update pending items.
7. Stop early if review reports no pending items and no blocker.

Important correction:

- The current author prompt tells the agent to complete as much work as
  possible in one session. It is not a strict “do exactly one task and exit”
  loop.

## In-memory cross-loop state

Cross-loop state lives in the `PromptLoopRunner` instance, not primarily in
files:

- `loop_summaries`
- `review_summaries`
- `pending_items`
- `loops_completed`

The harness injects these summaries back into later prompts.

## Persistent artifacts

The loop writes artifacts under `.codescribe/loop/`:

- `run.toml`
  - run metadata
  - configured model and limits
  - cumulative token counters
  - `loops_completed`
- `state.toml`
  - current `run_id`, `loop_index`, and `phase`
  - `workdir`
  - `task_file`
  - `updated_at`
- `author.toml`
  - author-phase event log for the most recent loop
  - overwritten at the start of each author phase
- `review_output.toml`
  - structured output from the review agent
  - overwritten on each review phase
- `review.toml`
  - review-phase event log
- `rules.toml`
  - active policy rules, resolved protected files, and rule parse problems
  - rewritten whenever rules change; written for inspection only, never read
    back (the in-memory rules are authoritative, so editing this file mid-run
    has no effect)
- `metadata/`
  - durable normalized per-phase telemetry files for plotting/comparison
    (tokens, iterations, wall time)
  - includes `manifest.toml` plus `loop_###_author.toml` / `loop_###_review.toml`

These files are useful for inspection and crash recovery, but the main state
relay is still in memory.

## Task loading and tool configuration

`PromptLoopRunner.__post_init__()`:

- resolves `workdir`,
- ensures the task file is inside the workdir,
- constructs the model with `set_neural_model(..., reasoning=reason)`,
- builds the loop system prompt with `build_system_prompt(...)`,
- loads the task chat template with `load_chat_template(..., return_meta=True)`,
- reads optional task metadata:

```toml
[tools]
bash = ["rg", "python", "python3"]
```

That `bash` list extends the bounded author-phase bash allowlist. An optional
`protect = [...]` list names files the agent may not modify; see
[Policy](#policy-protected-files-and-binding-rules).

If `bash` includes a python interpreter, python may only run `.py` files under
one directory, so every snippet the author runs is kept on disk:

```toml
[tools]
bash = ["python3"]
scripts_dir = "dev/transformations/x/scripts"   # default: .codescribe/tmp
```

`python3 -c ...`, `python3 -m ...`, stdin and the REPL are refused, as is any
script outside `scripts_dir`. If `scripts_dir` is not set, scripts go in
`.codescribe/tmp` (never anywhere else). The author prompt names this directory
and asks the author to write a check script there instead of running
`python -m compileall .`. See [tools.md](tools.md#python-runs-only-from-a-scripts-directory).

## System prompt

`build_system_prompt(...)` gives both loop phases a shared policy prompt.

It emphasizes:

- never fabricating file contents, command output, or test results,
- determining state by tools rather than assumption,
- staying inside the working directory,
- using relative or workdir-contained paths in bash,
- batching independent reads early,
- preferring direct implementation over repeated exploration,
- validating after meaningful changes,
- using `edit` for targeted changes and `write` for new files or rewrites.

## Author phase

The author phase uses:

- `Agent(...)`
- `tools = make_tools(workdir, bash_allow=...)`
- `max_iterations = agent_iterations`
- logging to `.codescribe/loop/author.toml`

If `--log` or `--log-path` is also provided, logs are fanned out through
`MultiToolLogSink` to both the author log and the requested extra log path.

### Author task prompt

`build_author_task(...)` injects:

- loop progress,
- files created across prior loops,
- files edited across prior loops,
- recent commands and errors from the last loop,
- pending items.

That context is assembled by `format_loop_context(...)`, which is how the
harness keeps the next author session oriented without requiring the agent to
re-read state files.

Behavior differs slightly on the first loop:

- if the task file can be read up front, its full contents are injected and the
  agent is told not to re-read it,
- otherwise the agent is told to read the task file.

Later loops are explicitly told not to re-read the task file or re-glob the
workspace just for orientation.

### Completion contract

The author agent is asked to finish with `<final_answer>` containing:

- `STATUS: COMPLETE` or `STATUS: INCOMPLETE`
- the plan followed
- exact check output
- when incomplete, a `NEXT STEPS:` section

The harness parses:

- `STATUS:` via `extract_status(...)`
- `NEXT STEPS:` via `extract_pending_items(...)`

## Review phase

The review phase runs a separate fresh agent.

Current review toolset:

- `read`
- `glob`
- `write`
- bounded `bash`

The review bash allowlist is tighter than the author phase:

- `ls`
- `stat`
- `pwd`
- `find`
- `grep`
- `head`
- `tail`
- `which`
- `env`
- `rg`

The review agent iteration budget is:

- `max(6, agent_iterations // 2)`

Important correction:

- The review phase is not limited to only `read/glob/write`; it also has a
  restricted `bash` tool.

### Review input

`build_review_task(...)` gives the review agent:

- a harness-computed summary of verified actions,
- any rejected tool calls,
- the author agent’s final report,
- the path where it must write `review_output.toml`.

Rejected calls are important: they were attempted by the author agent but the
harness refused to run them, so they had no workspace effect and should be
considered unverified.

### Review output format

The review agent is instructed to write TOML like:

```toml
loop = 2
summary = "What actually happened"
blocker = ""

[[pending]]
item = "Concrete next step"
```

It may also emit `[[rule]]` entries and a top-level `retire` list; see
[Policy](#policy-protected-files-and-binding-rules).

The harness then reads `pending`, `blocker`, `rule`, and `retire` from that file.

## Loop summaries

`loop_summary_from_result(...)` builds a `LoopSummary` from the structured
`RunResult` rather than reparsing the event log.

It tracks:

- `files_written`
- `files_edited`
- `files_read`
- `commands_run`
- `errors`
- `rejected`
- token counts

Important correction:

- Although older comments may refer to the TOML event log, the current summary
  logic is driven by `Agent.run()` output semantics.

## Early exit conditions

The loop stops early in either of these cases:

1. The author agent emits `STATUS: COMPLETE`.
2. The review output contains:
   - no pending items, and
   - an empty blocker.

## Safety model

Loop mode is bounded to the configured working directory.

Current enforcement includes:

- task file must be inside `workdir`,
- file tools resolve paths within the root,
- bash runs with a bounded allowlist,
- review bash is even more restrictive,
- protected files and binding rules (below).

This is a policy layer, not an OS sandbox.

## Policy: protected files and binding rules

Prompt instructions alone do not hold for weaker models: an author told to
"finish everything" will route around a human-approval gate if a tool lets it.
The policy layer moves those constraints into the harness so they hold
regardless of which model runs the loop.

### Protected files

The only task-file setting is one key in the existing `[tools]` table:

```toml
[tools]
bash = ["python3"]
protect = ["dev/transformations/x/approvals.toml", "dev/transformations/x/*.md"]
```

Entries are workdir-relative paths or globs. The task file itself is always
protected. Like `bash`, `protect` is read once at startup and never re-read,
because the task file sits inside the agent's writable root.


- `write` and `edit` refuse protected paths, and bash refuses them as arguments
  (author and review phases).
- An allowlisted interpreter such as `python3` can still write anything, so the
  harness snapshots protected files at the start of each author phase and
  compares them after every author bash call and at phase end. Changes are
  reverted (created files are removed), the bash result becomes an `Error:`
  naming the files, and the change is recorded as a violation.
- Violations are shown to the reviewer (as a blocker) and to the next author.
- Humans should edit protected files (e.g. record approvals) while the loop is
  stopped: the guard cannot tell a human's edit during an author phase from the
  agent's, and will revert it.

### Rules

Each rule has `text` and optionally `deny_bash` (regex) and `protect` (path or
glob). Every active rule is injected into the author task under
`HARNESS-ENFORCED POLICY`, ahead of the pending steps it overrides. Enforcement
differs by field:

- `text` is guidance only: compliance depends on the model.
- `deny_bash` is enforced: matching author bash commands are refused with
  `Error: command refused by rule Rn`.
- `protect` is enforced: the path joins the protected set.

Rules are not written in the task file. The review agent adds them on the fly,
turning each author deviation it sees into a rule, through `review_output.toml`:

```toml
retire = ["R5"]          # optional; review rules that no longer apply

[[rule]]
text = "Do not rewrite agent_log.md wholesale; append with edit."
protect = ""
deny_bash = ""
```

Safeguards against a weak reviewer:

- `[tools].protect` paths cannot be unprotected by any rule,
- protections only accumulate during a run (retiring a rule does not unprotect
  its path),
- invalid regexes and patterns that match every command are dropped (the rule
  text is kept) and recorded under `problems` in `rules.toml`,
- duplicate rule text is ignored, and at most 12 rules stay active
  (the oldest are dropped first).

The author prompt also treats a task-file stop condition or a binding rule as a
hard blocker: the author reports `STATUS: INCOMPLETE` instead of working around it.

## Defaults currently exposed by the CLI

`code-scribe loop` defaults today:

- `agent_loops = 5`
- `agent_iterations = 30`

## CLI note

The loop implementation and CLI option defaults agree on the current limits.

However, the command help text in `codescribe/cli/_commands.py` still contains
older wording that says each loop picks the single most important next task and
exits. That description is stale relative to the current author prompt in
`codescribe/lib/_loop.py`, which instructs the agent to complete as much work as
possible in one session.
