# Command layer

This page documents the current command implementations in `codescribe/lib/_cmd.py`
and how they are exposed through the CLI.

## CLI commands

The top-level Click group is `code-scribe`.

Current subcommands in `codescribe/cli/_commands.py`:

- `index`
- `draft`
- `translate`
- `generate`
- `update`
- `inspect`
- `format`
- `agent`
- `loop`

`codescribe/api/_commands.py` is a thin wrapper layer over these library entry
points.

## `prompt_translate`

File: `codescribe/lib/_cmd.py`

Purpose:

- Perform prompt-based Fortran-to-C++ translation.

Current behavior:

- Accepts a six-list mapping produced by `create_src_mapping(...)`.
- Loads the TOML chat template from `seed_prompt`.
- For each source tuple:
  - skips work if the C++ output file already exists,
  - strips Fortran comment lines before prompt injection,
  - appends `<source>...</source>` to the last user turn,
  - optionally appends `<draft>...</draft>` if a `.scribe` draft exists,
  - calls `model.chat(...)`,
  - extracts `<csource>`, `<cheader>`, and `<fsource>` blocks from the reply.
- Restores the last user message after each file so the seed prompt does not
  accumulate per-file state.

This is a prompt-driven workflow, not an agentic one.

## `prompt_inspect`

Purpose:

- Run a bounded read-only agent over a set of files and answer a query.

Current behavior:

- Requires `model`.
- Resolves all requested paths and computes a common bounded root.
- Builds a task listing the files and the query.
- Uses `lib.make_readonly_tools(common_root)`.

Important correction:

- `make_readonly_tools(...)` currently includes `read`, `glob`, and bounded
  `bash`. It is read-only with respect to files, but not bash-free.

The final answer is printed to stdout.

## `prompt_generate`

Purpose:

- Generate one or more files from a prompt.

Current behavior:

- Accepts either:
  - a path to a TOML chat template, or
  - a plain prompt string.
- Prepends a system instruction requiring XML-style tagged file output.
- Optionally injects reference files as read-only context.
- Calls `model.chat(...)`.
- Extracts any `<path>...</path>` pairs with a regex.
- Writes each matched file to disk, creating directories as needed.

## `prompt_update`

Purpose:

- Update one or more existing files from either a seed prompt or a natural
  language query.

Current behavior:

- Requires at least one of `seed_prompt` or `query_prompt`.
- Builds the same XML-tag output protocol used by `prompt_generate`.
- Injects target files as editable context.
- Injects `reference_existing` files as read-only context.
- Rejects overlap between target and reference files.
- Calls `model.chat(...)` and writes tagged outputs back to disk.

## `prompt_agent`

Purpose:

- Run a single tool-using agent session on a task.

Current behavior:

- Takes either a `task` string or a `task_file`. The two are mutually
  exclusive and exactly one is required.
- Constructs the model with `set_neural_model(model, reasoning=reason)`.
- Optionally enables TOML event logging via `ToolLogToml`.
- Uses `lib.make_tools(root, bash_allow=..., protected_paths=...)`, where
  `root` is `workdir` or the current working directory.
- Runs `Agent(...).run(task, chat_history=...)` and returns `str(result)`.

Task file handling (`load_agent_task_file`):

- The file must live inside the bounded root; otherwise `ensure_within_workdir`
  raises.
- It is read with `load_chat_template(..., return_meta=True)` — the same loader
  the loop command uses.
- The parsed chat template becomes the agent's prior conversation.
- The `[tools] bash = [...]` section extends the bounded bash allowlist.
- The file itself becomes a protected path, so `write`, `edit`, and `bash`
  may not target it.
- The task string passed to the agent is then built by `build_agent_task(...)`,
  which points at the task file and tells the agent the specification is
  already in context.

Important correction:

- The current implementation uses the bounded toolset rooted at the current
  working directory. It is not an unbounded filesystem agent.

CLI:

- `TASK` is either a task string or a path to a TOML task file. A path to an
  existing file is treated as a task file; a `.toml` argument that does not
  exist is rejected.

CLI flags:

- `--workdir`
- `--agent-iterations / -niter`
- `--verbose / -v`
- `--log`
- `--log-path`
- `--reason`

## `prompt_loop`

`loop` is implemented in `codescribe/lib/_loop.py`, not `_cmd.py`.
It is still part of the public command layer and is exposed through:

- `codescribe/cli/_commands.py: loop()`
- `codescribe/api/_commands.py: loop()`
- `codescribe/lib/_loop.py: prompt_loop()`

See [`loop.md`](./loop.md) for behavior details.

## Output conventions worth knowing

### Tagged-file output

`prompt_generate`, `prompt_update`, and parts of `prompt_translate` depend on
XML-style tags in model output, for example:

```text
<path/to/file.py>
print("hello")
</path/to/file.py>
```

### Agent return value

`Agent.run()` returns a `RunResult`. `prompt_agent()` converts that to a string
before returning it to the CLI, so CLI users see either:

- the final text, or
- a stop message such as max-iteration or tool-budget exhaustion.
