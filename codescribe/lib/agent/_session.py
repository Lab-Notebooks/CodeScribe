# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Single agent session: `code-scribe agent`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from codescribe.lib.util import ToolLogToml, load_chat_template
from codescribe.lib.models import set_neural_model
from codescribe.lib.tools import ensure_within_workdir, make_tools
from codescribe.lib.agent._agent import Agent


def load_agent_task_file(task_file: Union[Path, str], workdir: Path):
    """Read a TOML task file (same format as `loop`).

    Returns (task_path, chat_history, bash_allow, memory). The task file must
    live inside `workdir`.
    """
    task_path = ensure_within_workdir(Path(task_file), workdir)
    chat_history, meta = load_chat_template(task_path, return_meta=True)
    tools_cfg = meta.get("tools") or {}
    return (
        task_path,
        chat_history,
        set(tools_cfg.get("bash") or []),
        tools_cfg.get("memory"),
    )


def build_agent_task(*, workdir: Path, task_rel: str) -> str:
    return (
        f"Working directory: {workdir}\n"
        f"Task file: {task_rel} (read-only — contains the full specification)\n\n"
        "The task specification has already been loaded into the conversation above — "
        "do NOT re-read the task file to orient yourself.\n\n"
        "Complete the specification autonomously — do NOT ask for confirmation. "
        "Inspect the current state of relevant files before editing them, and after "
        "each meaningful change run the closest available check (tests, lint, typecheck).\n\n"
        "Finish with <final_answer> summarising what you changed and the exact "
        "output of the checks you ran.\n"
    )


def prompt_agent(
    task: str,
    model: Union[Path, str],
    agent_iterations: int = 20,
    verbose: bool = False,
    logging: Optional[Union[Path, str]] = None,
    reason: bool = False,
    task_file: Optional[Union[Path, str]] = None,
    workdir: Optional[Union[Path, str]] = None,
) -> str:
    """Run one bounded agent session on a task string or a task file.

    A task file's chat template becomes prior conversation, its [tools]
    section extends the bash allowlist, and the file itself is protected.
    `logging=""` writes events to the default log path.
    """
    task = task or ""
    if task.strip() and task_file is not None:
        raise ValueError("prompt_agent accepts either a task string or a task file")
    if not task.strip() and task_file is None:
        raise ValueError("prompt_agent requires either a task string or a task file")

    root = Path(workdir).resolve() if workdir else Path.cwd().resolve()
    chat_history, bash_allow, protected, memory = None, set(), set(), None
    if task_file is not None:
        task_path, chat_history, bash_allow, memory = load_agent_task_file(
            task_file, root
        )
        protected = {task_path}
        task = build_agent_task(workdir=root, task_rel=str(task_path.relative_to(root)))

    agent = Agent(
        set_neural_model(model, reasoning=reason),
        tools=make_tools(
            root, bash_allow=bash_allow, protected_paths=protected, memory=memory
        ),
        max_iterations=agent_iterations,
        show_diagnostics=verbose,
        logging=None if logging is None else ToolLogToml(path=str(logging) or None),
    )
    return str(agent.run(task, chat_history=chat_history))
