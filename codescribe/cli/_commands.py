# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Command-line commands for CodeScribe."""

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import click

from codescribe.cli import code_scribe
from codescribe import api


def _options(*decorators):
    """Apply several click decorators as one."""

    def apply(f):
        for d in reversed(decorators):
            f = d(f)
        return f

    return apply


model_option = click.option(
    "--model",
    "-m",
    required=True,
    envvar="CODESCRIBE_MODEL",
    show_envvar=True,
    help="Gen AI model name or path",
)
verbose_option = click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Print agent diagnostics (per-iteration reasoning and tool calls) to stdout",
)
reference_option = click.option(
    "--reference-existing",
    "-r",
    type=click.Path(exists=True),
    multiple=True,
    help="List of reference files",
)
workdir_option = click.option(
    "--workdir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Working directory bound for the agent; defaults to the current directory",
)
agent_options = _options(
    click.option(
        "--log",
        "log_enabled",
        is_flag=True,
        help="Write agent diagnostic events (TOML) to the default path: .codescribe/logs/toolusage.toml.",
    ),
    click.option(
        "--log-path",
        default=None,
        type=click.Path(dir_okay=True, file_okay=True, writable=True),
        help="Write agent diagnostic events (TOML) to PATH (implies --log).",
    ),
    click.option(
        "--reason",
        is_flag=True,
        help=(
            "Enable extended reasoning (adaptive thinking on Anthropic models, "
            "high reasoning effort on OpenAI-compatible models)."
        ),
    ),
)


def _paths(values: Iterable[Union[str, Path]]) -> List[Path]:
    return [Path(v) for v in values]


def _log_target(log_enabled: bool, log_path: Optional[str]) -> Optional[str]:
    """PATH for --log-path, "" (default path) for --log, else None."""
    if log_path is not None:
        return log_path
    return "" if log_enabled else None


def _split_task(task: str) -> Tuple[str, Optional[Path]]:
    """An existing file is a TOML task file; anything else is a task string."""
    candidate = Path(task)
    if candidate.is_file():
        return "", candidate
    if candidate.suffix == ".toml":
        raise click.UsageError(f"Task file '{task}' does not exist")
    if not task.strip():
        raise click.UsageError("Please provide a task string or a task file")
    return task, None


@code_scribe.command(name="index")
@click.argument("root-dir", required=True, type=click.Path(exists=True))
def index(root_dir: Path) -> None:
    """
    \b
    Index Fortran files along a project directory tree
    \b

    \b
    This command walks along the directory tree and
    parses files to create a mapping for modules, subroutines,
    and functions
    \b
    """
    click.echo(api.index(Path(os.path.abspath(root_dir))))


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
    api.draft(_paths(fortran_files))


@code_scribe.command(name="translate")
@click.argument("fortran-files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--seed-prompt", "-p", required=True, help="TOML seed file for chat template"
)
@model_option
def translate(fortran_files: List[Path], seed_prompt: Path, model: str) -> None:
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
    api.translate(_paths(fortran_files), Path(seed_prompt), model)


@code_scribe.command(name="generate")
@click.argument("seed-query-prompt", required=True)
@model_option
@reference_option
def generate(
    seed_query_prompt: str, model: str, reference_existing: List[Path]
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
    api.generate(seed_query_prompt, model, _paths(reference_existing))


@code_scribe.command(name="update")
@click.argument("filelist", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--seed-prompt", "-p", help="TOML seed file containing prompts")
@click.option("--query-prompt", "-q", help="Natural language prompt")
@model_option
@reference_option
def update(
    filelist: List[Path],
    seed_prompt: Optional[str],
    query_prompt: Optional[str],
    model: str,
    reference_existing: List[Path],
) -> None:
    """
    \b
    Perform AI based code update on files
    \b

    \b
    This command applies generative AI to update code
    based on specifications given in the prompt
    \b
    """
    if not seed_prompt and not query_prompt:
        raise click.UsageError(
            "Please provide either the '--seed-prompt/-p' or '--query-prompt/-q'"
        )
    api.update(
        _paths(filelist), model, seed_prompt, query_prompt, _paths(reference_existing)
    )


@code_scribe.command(name="inspect")
@click.argument("fortran-files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--query-prompt", "-q", required=True, help="Query prompt")
@model_option
@verbose_option
def inspect(
    fortran_files: List[Path], query_prompt: str, model: str, verbose: bool
) -> None:
    """
    \b
    Perform AI code inspection on files
    \b

    \b
    This command uses the agent in bounded read-only mode to inspect
    a list of files and answer a query. Results may vary based
    on the combination of files
    \b
    """
    api.inspect(_paths(fortran_files), query_prompt, model, verbose=verbose)


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
    api.format(_paths(seed_prompt_list))


@code_scribe.command(name="agent")
@click.argument("task", required=True)
@model_option
@workdir_option
@click.option(
    "--agent-iterations",
    "-niter",
    default=20,
    show_default=True,
    help="Maximum number of tool-call iterations",
)
@verbose_option
@agent_options
def agent(
    task: str,
    model: str,
    workdir: Optional[str],
    agent_iterations: int,
    verbose: bool,
    log_enabled: bool,
    log_path: Optional[str],
    reason: bool,
) -> None:
    """
    \b
    Run an autonomous agent on a task
    \b

    \b
    This command drives a generative AI model through an
    iterative tool-call loop until the task is complete.
    Available tools: read, glob, bash, edit, write
    \b

    \b
    TASK is either a natural language task string, or the path
    to a TOML task file in the same format used by the 'loop'
    command. The two forms are mutually exclusive
    \b
    """
    task_string, task_file = _split_task(task)
    click.echo(
        api.agent(
            task_string,
            model,
            agent_iterations=agent_iterations,
            verbose=verbose,
            logging=_log_target(log_enabled, log_path),
            reason=reason,
            task_file=task_file,
            workdir=Path(workdir) if workdir else None,
        )
    )


@code_scribe.command(name="loop")
@click.argument("task-file", required=True, type=click.Path(exists=True))
@model_option
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
@workdir_option
@verbose_option
@agent_options
def loop(
    task_file: str,
    model: str,
    agent_loops: int,
    agent_iterations: int,
    workdir: Optional[str],
    verbose: bool,
    log_enabled: bool,
    log_path: Optional[str],
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
    click.echo(
        api.loop(
            task_file=Path(task_file),
            model=model,
            agent_loops=agent_loops,
            agent_iterations=agent_iterations,
            verbose=verbose,
            logging=_log_target(log_enabled, log_path),
            workdir=Path(workdir) if workdir else None,
            reason=reason,
        )
    )
