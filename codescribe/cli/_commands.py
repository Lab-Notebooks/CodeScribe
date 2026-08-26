# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Command-line commands for CodeScribe."""

# Standard libraries
import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

# Feature libraries
import click

from codescribe.cli import code_scribe
from codescribe import api
from codescribe import lib


def _require_model(model: Optional[Union[str, Path]]) -> Union[str, Path]:
    if not model:
        raise click.UsageError(
            "Please provide the '--model/-m' option (or set CODESCRIBE_MODEL)"
        )
    return model


def _to_paths(values: Iterable[Union[str, Path]]) -> List[Path]:
    return [Path(value) for value in values]


def _split_task_argument(task: str) -> tuple:
    """Resolve the TASK argument into either a task string or a task file.

    An argument naming an existing file is treated as a TOML task file; anything
    else is treated as a natural language task string.
    """
    candidate = Path(task)

    if candidate.is_file():
        return "", candidate

    if candidate.suffix == ".toml":
        raise click.UsageError(f"Task file '{task}' does not exist")

    if not task.strip():
        raise click.UsageError("Please provide a task string or a task file")

    return task, None


def _resolve_logging(log_enabled: bool, log_path: Optional[str]) -> Optional[str]:
    if log_path is not None:
        return log_path
    if log_enabled:
        # Empty string means "use default log path" in ToolLogToml.
        return ""
    return None


@code_scribe.command(name="index")
@click.argument("root-dir", required=True, type=click.Path(exists=True))
def index(root_dir: Path) -> None:
    """
    \b
    Index Fortran files along a project directory tree
    \b

    \b
    This command walks along the directory directory tree and
    parses files to creating mapping for modules, subroutines,
    and functions
    \b
    """
    message: str = api.index(Path(os.path.abspath(root_dir)))
    click.echo(message)


@code_scribe.command(name="draft")
@click.argument("fortran-files", nargs=-1, required=True, type=click.Path(exists=True))
def draft(fortran_files: List[Path]) -> None:
    """
    \b
    Perform a draft conversion from Fortran to C++
    \b

    \b
    This command performs a line-by-line conversion to
    prepare a list of files for generative AI use
    \b
    """
    api.draft(_to_paths(fortran_files))


@code_scribe.command(name="translate")
@click.argument("fortran-files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--seed-prompt", "-p", required=True, help="TOML seed file for chat template"
)
@click.option(
    "--model",
    "-m",
    required=False,
    default=os.getenv("CODESCRIBE_MODEL"),
    help="Gen AI model name or path",
)
def translate(
    fortran_files: List[Path],
    seed_prompt: Path,
    model: Union[str, Path],
) -> None:
    """
    \b
    Perform AI based code conversion of Fortran files
    \b

    \b
    This command applies generative AI to convert code from
    Fortran to C++, and create a corresponding Fortran/C
    interface
    \b
    """
    api.translate(
        _to_paths(fortran_files),
        Path(seed_prompt),
        _require_model(model),
    )


@code_scribe.command(name="generate")
@click.argument("seed-query-prompt", required=True)
@click.option(
    "--model",
    "-m",
    required=False,
    default=os.getenv("CODESCRIBE_MODEL"),
    help="Gen AI model name or path",
)
@click.option(
    "--reference-existing",
    "-r",
    type=click.Path(exists=True),
    multiple=True,
    help="List of reference files",
)
def generate(
    seed_query_prompt: Union[Path, str],
    model: Union[Path, str],
    reference_existing: List[Path],
) -> None:
    """
    \b
    Perform AI based code generation
    \b

    \b
    This command applies generative AI to generate code
    based on specifications given in the prompt
    \b
    """
    api.generate(
        seed_query_prompt,
        _require_model(model),
        _to_paths(reference_existing),
    )


@code_scribe.command(name="update")
@click.argument("filelist", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--seed-prompt",
    "-p",
    help="TOML seed file containing prompts",
)
@click.option(
    "--query-prompt",
    "-q",
    help="Natural language prompt",
)
@click.option(
    "--model",
    "-m",
    required=True,
    default=os.getenv("CODESCRIBE_MODEL"),
    help="Gen AI model name or path",
)
@click.option(
    "--reference-existing",
    "-r",
    type=click.Path(exists=True),
    multiple=True,
    help="List of reference files",
)
def update(
    filelist: List[Path],
    seed_prompt: Path,
    query_prompt: str,
    model: Union[Path, str],
    reference_existing: List[Path],
) -> None:
    """
    \b
    Perform AI based code update on files
    \b

    \b
    This command applies generative AI to generate code
    based on specifications given in the prompt/
    \b
    """
    if (not seed_prompt) and (not query_prompt):
        raise click.UsageError(
            "Please provide either the '--seed-prompt/-p' or '--query-prompt/-q'"
        )

    api.update(
        _to_paths(filelist),
        model,
        seed_prompt,
        query_prompt,
        _to_paths(reference_existing),
    )


@code_scribe.command(name="inspect")
@click.argument("fortran-files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--query-prompt", "-q", required=True, help="Query prompt")
@click.option(
    "--model",
    "-m",
    required=False,
    default=os.getenv("CODESCRIBE_MODEL"),
    help="Gen AI model name or path",
)
@click.option(
    "--verbose",
    "-v",
    "verbose",
    is_flag=True,
    help="Print agent diagnostics (per-iteration reasoning and tool calls) to stdout",
)
def inspect(
    fortran_files: List[Path],
    query_prompt: str,
    model: Union[str, Path],
    verbose: bool,
) -> None:
    """
    \b
    Perform AI code inspection on files
    \b

    \b
    This command uses the agent in bounded read-only mode to inspect
    a list of files and answer a query. Results may vary based
    on the the combination of files
    \b
    """
    api.inspect(
        _to_paths(fortran_files),
        query_prompt,
        _require_model(model),
        verbose=verbose,
    )


@code_scribe.command(name="format")
@click.argument(
    "seed-prompt-list", nargs=-1, required=True, type=click.Path(exists=True)
)
def format(seed_prompt_list: List[Path]) -> None:
    """
    \b
    Format TOML seed prompt files
    \b

    \b
    This command loads the TOML prompt files
    and formats its contents into a markdown
    format
    \b
    """
    api.format(_to_paths(seed_prompt_list))


@code_scribe.command(name="agent")
@click.argument("task", required=True)
@click.option(
    "--model",
    "-m",
    required=True,
    default=os.getenv("CODESCRIBE_MODEL"),
    help="Gen AI model name or path",
)
@click.option(
    "--workdir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Working directory bound for the agent; defaults to the current directory",
)
@click.option(
    "--agent-iterations",
    "-niter",
    default=20,
    show_default=True,
    help="Maximum number of tool-call iterations",
)
@click.option(
    "--verbose",
    "-v",
    "verbose",
    is_flag=True,
    help="Print agent diagnostics (per-iteration reasoning and tool calls) to stdout",
)
@click.option(
    "--log",
    "log_enabled",
    is_flag=True,
    help=(
        "Write agent diagnostic events (TOML) to the default path: "
        ".codescribe/logs/toolusage.toml."
    ),
)
@click.option(
    "--log-path",
    "log_path",
    required=False,
    default=None,
    type=click.Path(dir_okay=True, file_okay=True, writable=True),
    help="Write agent diagnostic events (TOML) to PATH (implies --log).",
)
@click.option(
    "--reason",
    "reason",
    is_flag=True,
    help=(
        "Enable extended reasoning (adaptive thinking on Anthropic models, "
        "high reasoning effort on OpenAI-compatible models)."
    ),
)
def agent(
    task: str,
    model: Union[str, Path],
    workdir: Union[str, None],
    agent_iterations: int,
    verbose: bool,
    log_enabled: bool,
    log_path: Union[str, None],
    reason: bool,
) -> None:
    """
    \b
    Run an autonomous agent on a task
    \b

    \b
    This command drives a generative AI model through an
    iterative tool-call loop until the task is complete.
    Available tools: read, bash, edit, write
    \b

    \b
    TASK is either a natural language task string, or the path
    to a TOML task file in the same format used by the 'loop'
    command. The two forms are mutually exclusive
    \b
    """
    task_string, task_file = _split_task_argument(task)

    result = api.agent(
        task_string,
        _require_model(model),
        agent_iterations=agent_iterations,
        verbose=verbose,
        logging=_resolve_logging(log_enabled, log_path),
        reason=reason,
        task_file=task_file,
        workdir=Path(workdir) if workdir else None,
    )
    click.echo(result)


@code_scribe.command(name="loop")
@click.argument("task-file", required=True, type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    required=True,
    default=os.getenv("CODESCRIBE_MODEL"),
    help="Gen AI model name or path",
)
@click.option(
    "--agent-loops",
    "-nloop",
    default=5,
    show_default=True,
    help="Maximum number of bounded agent loops",
)
@click.option(
    "--agent-iterations",
    "-niter",
    default=30,
    show_default=True,
    help="Maximum tool-call iterations per agent session",
)
@click.option(
    "--workdir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Working directory bound for the agent; defaults to the current directory",
)
@click.option(
    "--verbose",
    "-v",
    "verbose",
    is_flag=True,
    help="Print agent diagnostics (per-iteration reasoning and tool calls) to stdout",
)
@click.option(
    "--log",
    "log_enabled",
    is_flag=True,
    help=(
        "Write agent diagnostic events (TOML) to the default path: "
        ".codescribe/logs/toolusage.toml."
    ),
)
@click.option(
    "--log-path",
    "log_path",
    required=False,
    default=None,
    type=click.Path(dir_okay=True, file_okay=True, writable=True),
    help="Write agent diagnostic events (TOML) to PATH (implies --log).",
)
@click.option(
    "--reason",
    "reason",
    is_flag=True,
    help=(
        "Enable extended reasoning (adaptive thinking on Anthropic models, "
        "high reasoning effort on OpenAI-compatible models)."
    ),
)
def loop(
    task_file: Path,
    model: Union[str, Path],
    agent_loops: int,
    agent_iterations: int,
    workdir: Union[str, None],
    verbose: bool,
    log_enabled: bool,
    log_path: Union[str, None],
    reason: bool,
) -> None:
    """
    \b
    Run a bounded agent loop
    \b

    \b
    Each loop runs a fresh bounded author session over the task file.
    The author agent tries to complete as much remaining work as possible
    in that session, and a separate review phase runs when needed.
    Cross-loop continuity is carried by harness-injected summaries.
    \b
    """
    result = api.loop(
        task_file=Path(task_file),
        model=_require_model(model),
        agent_loops=agent_loops,
        agent_iterations=agent_iterations,
        verbose=verbose,
        logging=_resolve_logging(log_enabled, log_path),
        workdir=Path(workdir) if workdir else None,
        reason=reason,
    )
    click.echo(result)
