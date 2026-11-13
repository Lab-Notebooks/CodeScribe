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
│  3. Harness records workspace diff, runs [check]       │
│                    │                                    │
│                    ├─→ COMPLETE + check passes? ──→ STOP│
│                    │                                    │
│  4. Build LoopSummary from author RunResult            │
│                         │                               │
│  5. ┌─────────────────────────────────────────────┐    │
│     │ REVIEW PHASE                                │    │
│     │ - Fresh Agent with read/glob/write/bash     │    │
│     │ - Judges diff + check against the spec      │    │
│     │ - Writes: review_output.toml                │    │
│     │ - Logs: .codescribe/loop/review.toml        │    │
│     └─────────────────────────────────────────────┘    │
│                    │                                    │
│                    ├─→ Clean review + check passes? → STOP│
│                    ├─→ No progress for 2 loops? ──→ STOP│
│                    └─→ Continue to next loop iteration  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

Cross-loop state carried in-memory: `loop_summaries`, `review_summaries`,
`items`, `blocker`, `scores`. Injected into next author phase task prompt.

## Current author/review model

For each loop iteration:

1. Reload task metadata if the task file changed.
2. Run an author-phase agent.
3. Record the workspace diff and run `[check]`.
4. If the author reported `STATUS: COMPLETE` and the check passed, stop.
5. Otherwise run a review-phase agent.
6. Read `review_output.toml`; update items, blocker and rules.
7. Score the loop; stop if complete or stalled.

Important correction:

- The current author prompt tells the agent to complete as much work as
  possible in one session. It is not a strict “do exactly one task and exit”
  loop.

## In-memory cross-loop state

Cross-loop state lives in the `PromptLoopRunner` instance, not primarily in
files:

- `loop_summaries`
- `review_summaries`
- `items` (open review items with stable ids)
- `blocker`
- `scores`
- `loops_completed`

The harness injects these summaries back into later prompts.

## Persistent artifacts

The loop writes artifacts under `.codescribe/loop/`:

- `run.toml`
  - run metadata
  - configured model and limits
  - cumulative token counters
  - `loops_completed`, `stop_reason` (`complete`, `stalled`, `max_loops`)
  - `[[score]]` per loop: `changed`, `check_passed`, `resolved`, `open_items`, `progress`
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
  - deleted before each review phase, so a missing write is never read as stale output
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
`protect = [...]` list names files the agent may not modify, and an optional
`writable = [...]` list names author-owned files no review rule may protect; see
[Policy](#policy-protected-files-and-binding-rules).

An optional `memory` key sets the memory layer (`mem` for short), a
workdir-relative directory for durable agent artifacts. If `bash` includes a
python interpreter, python may only run `.py` files under `<mem>/scripts`, so
every snippet the author runs is kept on disk:

```toml
[tools]
bash = ["python3"]
memory = "dev/transformations/x/mem"   # default: .codescribe/mem
```

`python3 -c ...`, `python3 -m ...`, stdin and the REPL are refused, as is any
script outside `<mem>/scripts`. The `scripts` subdirectory is fixed; only
`memory` is configurable. If `memory` is not set, scripts go in
`.codescribe/mem/scripts` (never anywhere else). The author prompt names this directory
and asks the author to write a check script there instead of running
`python -m compileall .`. See [tools.md](tools.md#python-runs-only-from-a-scripts-directory).

## Convergence

The loop is an optimization: the author proposes, the harness measures, the
reviewer critiques. Convergence is decided by the harness, not by agent
self-report.

### Check

```toml
[check]
command = "pytest -q"   # argv, no shell; exit 0 means pass
timeout = 600           # seconds (default 600)
```

The harness runs the check after every author phase. Its verdict and output
tail go to the reviewer (always) and to the next author (on failure). Like
`[tools]`, `[check]` is read once at startup. Without a check, completion rests
on a clean review.

### Feedback

- The reviewer gets the phase's workspace diff, taken from git trees built with
  a throwaway index (untracked files included, `.codescribe/` excluded, the
  user's staging area untouched). Capped at 20000 chars. Outside git, no diff.
- The next author gets the check result, the reviewer's `blocker`, and the open
  review items with their ids.
- Both phases see the task's chat as conversation; the spec is never injected twice.

### Items

Review items carry stable ids (`P1`, `P2`, …). The reviewer restates an open
item with its `id` to keep it open; open items it omits are resolved; items
without a known id are new. At most 8 stay open.

### Stopping

- `complete`: author `STATUS: COMPLETE` with a passing check, or a clean review
  (no items, no blocker) with a passing or absent check.
- `stalled`: `stall_loops` (default 2) consecutive loops without progress.
  Progress means the workspace changed and an item was resolved or the check
  turned green.
- `max_loops`: `agent_loops` exhausted.

## System prompt

`build_system_prompt(...)` is the author's policy prompt;
`build_review_system_prompt()` is the reviewer's.

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
- the last check result and errors,
- the reviewer's blocker and open review items,
- the bash allowlist.

That context is assembled by `format_loop_context(...)`, which is how the
harness keeps the next author session oriented without requiring the agent to
re-read state files.

The task's chat is passed as conversation, so the agent is told not to re-read
the task file or re-glob the workspace just for orientation.

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

- the task's chat as conversation,
- a harness-computed summary of verified actions,
- the check result and workspace diff,
- any rejected tool calls,
- open review items from earlier loops,
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
id = "P1"            # omit for a new item
item = "Concrete defect or missing work"
```

It may also emit `[[rule]]` entries and a top-level `retire` list; see
[Policy](#policy-protected-files-and-binding-rules).

The harness then reads `pending`, `blocker`, `rule`, and `retire` from that file.
If the file is missing, the author's `NEXT STEPS` become the open items.

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

See [Stopping](#stopping).

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

The task-file settings are two keys in the existing `[tools]` table:

```toml
[tools]
bash = ["python3"]
protect = ["dev/transformations/x/approvals.toml", "dev/transformations/x/*_spec.md"]
writable = ["dev/transformations/x/agent_log.md"]
```

Entries are workdir-relative paths or globs. The task file itself is always
protected. `writable` only limits what review rules may protect; it does not
override `protect`. Like `bash`, `protect` and `writable` are read once at
startup and never re-read, because the task file sits inside the agent's
writable root.


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
- `protect` is enforced: the path joins the protected set while the rule is
  active.

Rules are not written in the task file. The review agent adds them on the fly,
for harmful or repeated author deviations, through `review_output.toml`:

```toml
retire = ["R5"]          # optional; review rules that no longer apply

[[rule]]
text = "Do not rewrite agent_log.md wholesale; append with edit."
protect = ""
deny_bash = ""
```

Safeguards against a weak reviewer:

- `[tools].protect` paths cannot be unprotected by any rule,
- a rule's `protect` cannot target a `[tools].writable` path: the protect is
  dropped (the rule text is kept), recorded under `problems`, and shown to the
  next review. Without this, a reviewer writing a rule *about* a file the author
  must append to (e.g. "log every retirement in agent_log.md") can set
  `protect` on it and lock the author out, stalling the run,
- a rule's protection lasts only while the rule is active, so retiring a rule
  undoes a protection that blocked required work,
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
