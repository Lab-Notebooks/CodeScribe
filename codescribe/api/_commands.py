# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Public API entry points for CodeScribe commands."""

from pathlib import Path
from typing import List, Optional, Union

from codescribe import lib


def index(root_dir: Path) -> str:
    """Index Fortran modules/subroutines/functions under `root_dir` into scribe.yaml."""
    lib.create_scribe_yaml(root_dir)
    return "Project structure saved to scribe.yaml."


def draft(fortran_files: List[Path]) -> None:
    """Write line-by-line draft annotations for Fortran files."""
    file_index = lib.create_file_indexes()
    for sfile in fortran_files:
        print(lib.annotate_fortran_file(sfile, file_index))


def translate(filelist: List[Path], seed_prompt: Path, model: Union[Path, str]) -> None:
    """Translate Fortran files to C++ with a Fortran/C interface."""
    lib.prompt_translate(lib.create_src_mapping(filelist), seed_prompt, model=model)


def inspect(
    filelist: List[Path],
    query_prompt: str,
    model: Union[Path, str],
    verbose: bool = False,
) -> None:
    """Answer a query about files with a read-only agent."""
    lib.prompt_inspect(filelist, query_prompt, {}, model=model, verbose=verbose)


def generate(
    seed_query_prompt: Union[Path, str],
    model: Union[Path, str],
    reference_existing: Optional[List[Path]] = None,
) -> None:
    """Generate code from a TOML seed prompt or a natural-language query."""
    lib.prompt_generate(
        seed_query_prompt, model=model, reference_existing=reference_existing or []
    )


def update(
    filelist: List[Path],
    model: Union[Path, str],
    seed_prompt: Optional[Path] = None,
    query_prompt: str = "",
    reference_existing: Optional[List[Path]] = None,
) -> None:
    """Update files from a seed prompt or a query."""
    if not seed_prompt and not query_prompt:
        raise ValueError("Please provide either the 'seed_prompt' or 'query_prompt'")
    lib.prompt_update(
        filelist,
        seed_prompt,
        query_prompt,
        model=model,
        reference_existing=reference_existing or [],
    )


def format(seed_prompt_list: List[Path]) -> None:
    """Reformat TOML prompt files."""
    for seed_prompt in seed_prompt_list:
        lib.format_seed_prompt(seed_prompt)


def agent(
    task: str = "",
    model: Union[Path, str] = "",
    agent_iterations: int = 20,
    verbose: bool = False,
    logging: Union[Path, str, None] = None,
    reason: bool = False,
    task_file: Union[Path, str, None] = None,
    workdir: Union[Path, None] = None,
) -> str:
    """Run one agent session on a task string or a TOML task file (exactly one)."""
    if not model:
        raise ValueError("Please provide the 'model'")
    return lib.prompt_agent(
        task,
        model=model,
        agent_iterations=agent_iterations,
        verbose=verbose,
        logging=logging,
        reason=reason,
        task_file=task_file,
        workdir=workdir,
    )


def loop(
    task_file: Path,
    model: Union[Path, str],
    agent_loops: int = 5,
    agent_iterations: int = 30,
    verbose: bool = False,
    logging: Union[Path, str, None] = None,
    workdir: Union[Path, None] = None,
    reason: bool = False,
) -> str:
    """Run the bounded author → review loop over a task file."""
    return lib.prompt_loop(
        task_file=task_file,
        model=model,
        agent_loops=agent_loops,
        agent_iterations=agent_iterations,
        verbose=verbose,
        logging=logging,
        workdir=workdir,
        reason=reason,
    )
