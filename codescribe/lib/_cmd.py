# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

import re
import os, json, textwrap

from typing import List, Dict, Union, Optional, Any
from pathlib import Path
from alive_progress import alive_bar

from codescribe import lib

__all__ = [
    "prompt_translate",
    "prompt_inspect",
    "prompt_generate",
    "prompt_update",
    "load_agent_task_file",
    "build_agent_task",
    "prompt_agent",
]


def prompt_translate(
    mapping: List[str],
    seed_prompt: Path,
    model: Union[Path, str] = None,
) -> None:
    """Perform translation using prompts and the supplied model."""
    neural_model = None

    if model:
        print("Starting neural conversion process")
        neural_model = lib.set_neural_model(model)

    chat_template = lib.load_chat_template(seed_prompt)

    with alive_bar(len(mapping[0]), bar="blocks") as bar:

        for fsource, csource, finterface, cdraft, promptfile, cheader in zip(
            mapping[0], mapping[1], mapping[2], mapping[3], mapping[4], mapping[5]
        ):

            bar.text(fsource)
            bar()

            if not os.path.isfile(csource):
                cached_prompt = chat_template[-1]["content"]

                with open(fsource, "r") as sfile:
                    is_comment = False
                    source_code = []

                    for line in sfile.readlines():
                        is_comment = False

                        if line.strip().lower().startswith(("c", "!!", "!")) and (
                            not line.strip().lower().startswith(("complex"))
                        ):
                            is_comment = True

                        if not is_comment:
                            source_code.append(line)

                    if source_code:
                        chat_template[-1]["content"] += (
                            "\n" + "<source>\n" + "".join(source_code) + "</source>"
                        )

                if os.path.isfile(cdraft):

                    draft_code = []
                    with open(cdraft) as dfile:
                        for line in dfile.readlines():
                            draft_code.append(line)

                        if draft_code:
                            chat_template[-1]["content"] += (
                                "\n\n" + "<draft>\n" + "".join(draft_code) + "</draft>"
                            )

                if neural_model:
                    result = neural_model.chat(chat_template)

                    with open(csource, "w") as cdest, open(
                        finterface, "w"
                    ) as fdest, open(cheader, "w") as chead:

                        cheader = re.search(
                            r"<cheader>(.*?)</cheader>", result, re.DOTALL
                        )
                        csource = re.search(
                            r"<csource>(.*?)</csource>", result, re.DOTALL
                        )
                        fsource = re.search(
                            r"<fsource>(.*?)</fsource>", result, re.DOTALL
                        )

                        if csource:
                            cdest.write(csource.group(1))
                        else:
                            cdest.write(result)

                        if cheader:
                            chead.write(cheader.group(1))
                        else:
                            chead.write(result)

                        if fsource:
                            fdest.write(fsource.group(1))

                # lib.write_archive_toml(
                #    chat_template + [{"role": "assistant", "content": result}],
                #    neural_model,
                # )

                chat_template[-1]["content"] = cached_prompt

            else:
                continue


def prompt_inspect(
    filelist: List[Path],
    query_prompt: str,
    file_index: Dict[str, str] = {},
    model: Union[Path, str] = None,
    verbose: bool = False,
) -> None:
    """Perform inspection on a list of files using the agent in bounded read-only mode."""
    if not model:
        raise ValueError("prompt_inspect requires a model when running in agent mode")

    print("Performing agent-based inspection")
    neural_model = lib.set_neural_model(model)

    resolved_files = [Path(f).resolve() for f in filelist]
    if not resolved_files:
        raise ValueError("prompt_inspect requires at least one file")

    # For common_root: directories contribute themselves; files contribute their parent dir.
    roots = [p if p.is_dir() else p.parent for p in resolved_files]
    common_root = Path(os.path.commonpath([str(p) for p in roots])).resolve()

    # Keep user-requested files/dirs "as-is" in the context list.
    rel_files = [str(p.relative_to(common_root)) for p in resolved_files]

    filtered_file_index = {}
    for fsource in resolved_files:
        if file_index:
            filtered_file_index.update(lib.filter_file_indexes(fsource, file_index))

    system = (
        "You are a coding assistant performing code inspection in bounded read-only mode. "
        "Use the available tools to inspect the requested files and answer the query. "
        "Do not claim to have read files unless you actually read them with tools. "
        "Do not modify files or simulate command output. "
        "Only access paths within the bounded working directory and prefer the listed files."
    )

    task = "Inspect the following files and answer the query.\n\n"
    task += "Files to inspect:\n"
    for path in rel_files:
        task += f"- {path}\n"

    if filtered_file_index:
        task += "\nProject index:\n"
        for construct, file_path in filtered_file_index.items():
            task += f"- {construct}: {file_path}\n"

    task += "\nQuery:\n"
    task += query_prompt + "\n\n"
    task += "Use read and bounded bash as needed, then finish with <final_answer> containing the answer."

    coding_agent = lib.Agent(
        neural_model,
        tools=lib.make_readonly_tools(common_root),
        max_iterations=20,
        show_diagnostics=verbose,
    )
    result = coding_agent.run(task, system=system)
    print(result.final_text or str(result))


def prompt_generate(
    seed_prompt: Union[Path, str],
    model: Union[Path, str] = None,
    reference_existing: List[Path] = [],
) -> None:
    """Perform code generation based on the provided seed prompt."""
    neural_model = None

    if model:
        print("Performing neural generation")
        neural_model = lib.set_neural_model(model)

    system_template = [{"role": "system", "content": ""}]
    system_template[-1]["content"] += (
        "You are a code generation and editing assistant.\n"
        + "When the user asks for code that spans multiple files,\n"
        + "output each file enclosed within\n"
        + "XML-style tags using the format:\n"
        + "\n"
        + "<filename1>\n"
        + "... file contents ...\n"
        + "</filename1>\n"
        + "\n"
        + "<filename2>\n"
        + "... file contents ...\n"
        + "</filename2>\n"
        + "\n"
        + "Do not add any explanations or commentary outside of these tags.\n"
        + "Note that some of these files may be requested to be treated as read-only.\n"
        + "Do not edit or generate files that are requested as read-only."
    )

    if os.path.exists(seed_prompt):
        chat_template = system_template + lib.load_chat_template(seed_prompt)
    elif isinstance(seed_prompt, str):
        chat_template = system_template + [{"role": "user", "content": seed_prompt}]
    else:
        raise ValueError(f"Cannot handle seed_prompt")

    if reference_existing:
        chat_template[-1]["content"] += (
            "\n\nUse the content of the following files as a reference to \n"
            + "update the files above. Do not edit the files below; treat them as read-only.\n\n"
        )

        for filename in reference_existing:
            with open(filename, "r") as sfile:
                source_code = []

                for line in sfile.readlines():
                    source_code.append(line)

            if source_code:
                chat_template[-1]["content"] += (
                    "\n" + f"<{filename}>\n" + "".join(source_code) + f"</{filename}>\n"
                )

    if neural_model:
        result = neural_model.chat(chat_template)

        pattern = re.compile(r"<([^>]+)>\s*(.*?)\s*</\1>", re.DOTALL)
        cwd_root = Path.cwd().resolve()

        for match in pattern.finditer(result):
            filename, content = match.groups()
            try:
                target = lib.AgentTool.resolve_within_root(cwd_root, filename)
            except ValueError as exc:
                print(f"Skipped {filename!r}: {exc}")
                continue
            os.makedirs(target.parent, exist_ok=True)
            with open(target, "w") as f:
                f.write(content.strip() + "\n")
            print(f"Wrote {target}")

        # lib.write_archive_toml(
        #    chat_template + [{"role": "assistant", "content": result}], neural_model
        # )


def prompt_update(
    filelist: List[Path],
    seed_prompt: Path,
    query_prompt: str,
    model: Union[Path, str] = None,
    reference_existing: List[Path] = [],
):
    """Perform code updates based on the provided seed prompt and file list."""
    neural_model = None

    print("Performing neural update")
    neural_model = lib.set_neural_model(model)

    system_template = [{"role": "system", "content": ""}]
    system_template[-1]["content"] += (
        "You are a code generation and editing assistant.\n"
        + "When the user asks for code that spans multiple files,\n"
        + "output each file enclosed within\n"
        + "XML-style tags using the format:\n"
        + "\n"
        + "<filename1>\n"
        + "... file contents ...\n"
        + "</filename1>\n"
        + "\n"
        + "<filename2>\n"
        + "... file contents ...\n"
        + "</filename2>\n"
        + "\n"
        + "Do not add any explanations or commentary outside of these tags.\n"
        + "Note that some of these files may be requested to be treated as \n"
        + "read-only or may not be appended.\n"
        + "Do not edit files if they are not appended or requested as read-only."
    )

    if seed_prompt:
        chat_template = system_template + lib.load_chat_template(seed_prompt)
    else:
        chat_template = system_template + [{"role": "user", "content": ""}]

    if query_prompt:
        chat_template[-1]["content"] += query_prompt

    if set(filelist) & set(reference_existing):
        raise ValueError("Reference and target files should be mutually exclusive")

    if filelist:
        chat_template[-1]["content"] += (
            "\n\nUpdate the content of the following files based on\n"
            + "the instructions. Enclose the output of each file in their\n"
            + "respective XML elements. Only update the following files.\n\n"
        )

        for filename in filelist:
            with open(filename, "r") as sfile:
                source_code = []

                for line in sfile.readlines():
                    source_code.append(line)

            if source_code:
                chat_template[-1]["content"] += (
                    "\n" + f"<{filename}>\n" + "".join(source_code) + f"</{filename}>\n"
                )

    if reference_existing:
        chat_template[-1]["content"] += (
            "Use the content of the following files as a reference to \n"
            + "update the files above. Do not edit the files below; treat them as read-only.\n\n"
        )

        for filename in reference_existing:
            with open(filename, "r") as sfile:
                source_code = []

                for line in sfile.readlines():
                    source_code.append(line)

            if source_code:
                chat_template[-1]["content"] += (
                    "\n" + f"<{filename}>\n" + "".join(source_code) + f"</{filename}>\n"
                )

    if neural_model:
        result = neural_model.chat(chat_template)

        pattern = re.compile(r"<([^>]+)>\s*(.*?)\s*</\1>", re.DOTALL)
        cwd_root = Path.cwd().resolve()
        allowed_targets = {Path(f).resolve() for f in filelist}

        for match in pattern.finditer(result):
            filename, content = match.groups()
            try:
                target = lib.AgentTool.resolve_within_root(cwd_root, filename)
            except ValueError as exc:
                print(f"Skipped {filename!r}: {exc}")
                continue
            if target not in allowed_targets:
                print(f"Skipped {filename!r}: not in the requested file list")
                continue
            os.makedirs(target.parent, exist_ok=True)
            with open(target, "w") as f:
                f.write(content.strip() + "\n")
            print(f"Wrote {target}")

        # lib.write_archive_toml(
        #    chat_template + [{"role": "assistant", "content": result}], neural_model
        # )


def load_agent_task_file(task_file: Union[Path, str], workdir: Path):
    """Load a TOML task file for a single agent session.

    Uses the same reader as the loop command: the chat template becomes the
    agent's prior conversation, and the optional `[tools] bash = [...]` section
    extends the bounded bash allowlist. The task file must live inside
    *workdir* and is returned so callers can protect it from writes.

    Returns (task_path, chat_history, bash_allow).
    """
    task_path = lib.ensure_within_workdir(Path(task_file), workdir)
    chat_history, meta = lib.load_chat_template(task_path, return_meta=True)
    bash_allow = set((meta.get("tools") or {}).get("bash") or [])
    return task_path, chat_history, bash_allow


def build_agent_task(*, workdir: Path, task_rel: str) -> str:
    """Build the agent task string for a session driven by a task file."""
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
    """Run the agentic loop on *task* using the supplied model string.

    The model string uses the same prefix conventions as the other prompt_*
    functions (e.g. "anthropic-claude-sonnet-4-6", "openai-gpt-4o", etc.).
    The Agent drives the model through iterative tool calls until it emits a
    <final_answer> block and returns that text.

    *task* and *task_file* are mutually exclusive; exactly one is required.
    When *task_file* is given it is read with the same loader the loop command
    uses: its chat template is injected as prior conversation, its `[tools]`
    section extends the bash allowlist, and the file itself is protected from
    write/edit/bash.

    Tools are bounded to *workdir* (default: the current working directory).

    Set verbose=True to print agent diagnostics (per-iteration reasoning and tool calls)
    to stdout as the agent works.
    """
    task = task or ""
    if task.strip() and task_file is not None:
        raise ValueError("prompt_agent accepts either a task string or a task file")
    if (not task.strip()) and task_file is None:
        raise ValueError("prompt_agent requires either a task string or a task file")

    root = Path(workdir).resolve() if workdir else Path.cwd().resolve()

    chat_history = None
    bash_allow = set()
    protected_paths = set()
    agent_task = task

    if task_file is not None:
        task_path, chat_history, bash_allow = load_agent_task_file(task_file, root)
        protected_paths = {task_path}
        agent_task = build_agent_task(
            workdir=root,
            task_rel=str(task_path.relative_to(root)),
        )

    neural_model = lib.set_neural_model(model, reasoning=reason)

    logfile = None
    if logging is not None:
        # If logging is passed as an empty string (e.g. CLI --log with no PATH),
        # ToolLogToml will use its default location.
        logfile = lib.ToolLogToml(path=str(logging) if str(logging) else None)

    # Always use bounded tools rooted at the working directory.
    tools = lib.make_tools(
        root,
        bash_allow=bash_allow,
        protected_paths=protected_paths,
    )

    coding_agent = lib.Agent(
        neural_model,
        tools=tools,
        max_iterations=agent_iterations,
        show_diagnostics=verbose,
        logging=logfile,
    )
    return str(coding_agent.run(agent_task, chat_history=chat_history))
