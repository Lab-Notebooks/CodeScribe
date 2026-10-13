# Tools

This page documents the current tool implementations in
`codescribe/lib/_tools.py`.

## Tool set

CodeScribe currently provides five tools:

- `read`
- `glob`
- `bash`
- `edit`
- `write`

All tools inherit from `AgentTool`.

## `AgentTool`

Base fields:

- `name`
- `description`
- `parameters`
- `enabled`

Base helpers:

- `run(args)`
- `to_openai_tool()`
- `describe_for_prompt()`
- `resolve_within_root(root, target)`

`resolve_within_root(...)` is the main path-bounding primitive used by file
operations.

## `read`

Class:

- `ReadTool`

Arguments:

- `path` required
- `offset` optional, 1-indexed
- `limit` optional
- `with_line_numbers` optional, defaults to `1`

Behavior:

- reads text files with optional slicing,
- returns a header containing the resolved path and line range when line numbers
  are enabled,
- prefixes each returned line with a stable line number.

When bounded, the file path must resolve within the configured root.

## `glob`

Class:

- `GlobTool`

Arguments:

- `pattern` required
- `root` optional
- `include_dirs` optional
- `limit` optional, default `2000`

Behavior:

- uses Python glob with recursive `**` support,
- returns newline-separated matches,
- returns paths relative to the selected search base when possible,
- filters out directories unless `include_dirs=1`.

When bounded, both the search root and returned matches are constrained to the
configured root.

## `bash`

Class:

- `BashTool`

Arguments:

- `command` required
- `timeout` optional

Return format:

```text
exit_code: 0
STDOUT:
...
STDERR:
...
```

### Bounded bash

When `bounded=True`, `validate_command(...)` currently:

- rejects commands containing any of these characters:
  - `| & ; > < \` $`
- parses the command with `shlex.split(...)`
- requires the executable name to appear in the allowlist.

Default allowlist:

- `ls`
- `pwd`
- `find`
- `grep`
- `head`
- `tail`
- `wc`
- `git`
- `test`
- `echo`
- `sed`

### Python runs only from a scripts directory

When the allowlist includes a python interpreter (`python`, `python3`,
`python3.11`, `ipython`, `pypy3`, ...), bounded bash only accepts

```
python3 [safe flags] <scripts_dir>/<name>.py [args...]
```

so every piece of python the agent executes is left on disk as a file, which
keeps a record of where results came from. `-c`, `-m`, `-` (stdin), a bare
`python3` (REPL), flags that take an argument (`-W`, `-X`, ...), non-`.py`
targets, scripts outside the directory (symlinks are resolved), and missing
files are all rejected with a hint to write the script first. Allowed flags:
`-u -B -I -E -s -S -O -OO -q -b -bb -P`.

`scripts_dir` comes from the task file's `[tools].scripts_dir`. If it is not
set, it falls back to `.codescribe/tmp` (`DEFAULT_SCRIPT_DIR`). It must
resolve inside the workdir.

The check only looks at how python is invoked. Other allowlisted launchers
(e.g. `env`, `uv`, `make`) can still start python some other way.

Important current detail:

- even bounded bash still runs through `subprocess.run(..., shell=True)`.
  Safety comes from pre-validation, not from shell removal or process sandboxing.

## `edit`

Class:

- `EditTool`

Arguments:

- `path`
- `edits`, a list of objects with:
  - `oldText`
  - `newText`

Behavior:

- reads the full original file,
- requires every `oldText` to match exactly once in the original content,
- applies replacements by repeated string replacement on the evolving output,
- writes the updated file,
- returns a JSON report with verification snippets.

Important current behavior:

- uniqueness is checked against the original file content for each requested
  edit, not incrementally after earlier edits.
- the tool itself does not explicitly detect overlapping or nested edit regions;
  it relies on exact-match uniqueness and agent discipline.

## `write`

Class:

- `WriteTool`

Arguments:

- `path`
- `content`

Behavior:

- creates parent directories as needed,
- creates or overwrites the target file,
- returns a short confirmation string including byte count.

When bounded, the target path must resolve within the configured root.

## Toolset constructors

### `make_tools(root, bash_allow=None, protected_paths=None, scripts_dir=None)`

Returns the default bounded toolset:

- `read`
- `glob`
- bounded `bash`
- `edit`
- `write`

`bash_allow` extends the default bounded-bash allowlist. If it includes a
python interpreter, python is restricted to scripts under `scripts_dir`
(default `.codescribe/tmp`); see above.

### `make_readonly_tools(root, bash_allow=None, protected_paths=None, scripts_dir=None)`

Returns the bounded read-only toolset:

- `read`
- `glob`
- bounded `bash`

Important correction:

- “read-only” here means no `edit` or `write`. It still includes bounded `bash`.

### `DEFAULT_TOOLS`

`DEFAULT_TOOLS` is built at import time as:

- `make_tools(Path.cwd().resolve())`

In practice, most command paths construct their own rooted toolsets rather than
relying directly on this module-level default.
