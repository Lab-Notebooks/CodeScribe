# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""_loop

Prompt-loop implementation with in-memory cross-loop state.

State relay design:
  - Within a loop (iteration → iteration): message history + WORKSPACE CONTEXT block.
  - Across loops (loop N → loop N+1): Python objects held in prompt_loop().
    The harness builds LoopSummary from the agent's structured RunResult after
    each execution and injects it directly into the next loop's task string.
    Agents never read state files to orient themselves.

Persistent files (for inspection / crash-resume only):
  run.toml        — run configuration
  state.toml      — current loop index + phase
  execution.toml  — raw event log for the most recent execution phase
  review_output.toml — structured output from the review agent (pending items)
"""

from __future__ import annotations

import os
import toml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from codescribe import lib

__all__ = [
    "LoopPaths",
    "get_loop_paths",
    "load_state",
    "write_state",
    "LoopSummary",
    "build_system_prompt",
    "build_execution_task",
    "build_review_task",
    "prompt_loop",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass
class LoopPaths:
    """Paths under `.codescribe/loop/` for a single shared run."""

    run_dir: Path
    run_toml: Path
    state_toml: Path
    execution_toml: Path       # raw event log (overwritten each execution phase)
    review_output_toml: Path   # review agent's structured output (overwritten each review)


def get_loop_paths(workdir: Path) -> LoopPaths:
    run_dir = workdir / ".codescribe" / "loop"
    return LoopPaths(
        run_dir=run_dir,
        run_toml=run_dir / "run.toml",
        state_toml=run_dir / "state.toml",
        execution_toml=run_dir / "execution.toml",
        review_output_toml=run_dir / "review_output.toml",
    )


# ---------------------------------------------------------------------------
# State persistence (TOML)
# ---------------------------------------------------------------------------

_VALID_PHASES = frozenset({"idle", "execution", "review"})


def load_state(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    data = lib.read_toml(path)
    return data or None


def write_state(path: Path, state: Dict[str, Any]) -> None:
    state = dict(state)
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"state['run_id'] must be a non-empty str, got {run_id!r}")
    loop_index = state.get("loop_index")
    if not isinstance(loop_index, int) or isinstance(loop_index, bool):
        raise ValueError(f"state['loop_index'] must be an int, got {loop_index!r}")
    phase = state.get("phase")
    if phase not in _VALID_PHASES:
        raise ValueError(f"state['phase'] must be one of {sorted(_VALID_PHASES)}, got {phase!r}")
    state["updated_at"] = lib.iso_utc_now()
    lib.atomic_write_toml(path, state)


def init_state(*, run_id: str, workdir: Path, task_file: Path) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "workdir": str(workdir),
        "task_file": str(task_file),
        "loop_index": 0,
        "phase": "idle",
        "updated_at": lib.iso_utc_now(),
    }


# ---------------------------------------------------------------------------
# In-memory cross-loop state
# ---------------------------------------------------------------------------

@dataclass
class LoopSummary:
    """Harness-computed summary of one execution phase.

    Built deterministically from the TOML event log — no LLM involved.
    Carried in-memory by prompt_loop() and injected into subsequent task prompts.
    """

    loop_index: int
    files_written: List[str] = field(default_factory=list)
    files_edited: List[str] = field(default_factory=list)
    files_read: List[str] = field(default_factory=list)
    commands_run: List[str] = field(default_factory=list)   # "cmd → exit_code N"
    errors: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)        # "tool(args): reason" — attempted, never executed
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0


@dataclass
class PromptLoopRunner:
    task_file: Union[Path, str]
    model: Union[Path, str]
    agent_loops: int = 5
    agent_iterations: int = 30
    verbose: bool = False
    logging: Optional[Union[Path, str]] = None
    workdir: Optional[Union[Path, str]] = None
    reason: bool = False

    def __post_init__(self) -> None:
        self.workdir_path = Path(self.workdir).resolve() if self.workdir else Path.cwd().resolve()
        self.task_path = ensure_within_workdir(Path(self.task_file), self.workdir_path)
        self.neural_model = lib.set_neural_model(self.model, reasoning=self.reason)  # type: ignore[attr-defined]

        self.run_id = lib.new_run_id()
        self.paths = get_loop_paths(self.workdir_path)
        self.task_rel = str(self.task_path.relative_to(self.workdir_path))
        self.review_output_rel = str(self.paths.review_output_toml.relative_to(self.workdir_path))
        self.system = build_system_prompt(workdir=self.workdir_path)

        self.loop_summaries: List[LoopSummary] = []
        self.review_summaries: List[LoopSummary] = []
        self.pending_items: List[str] = []
        self.loops_completed = 0
        self.task_mtime: float = 0.0

        self.load_task_context()
        self.initialize_paths()
        self.initialize_run_record()
        self.initialize_loop_state()

    def load_task_context(self) -> None:
        self.chat_history, meta = lib.load_chat_template(self.task_path, return_meta=True)
        bash_allow = set((meta.get("tools") or {}).get("bash") or [])
        self.tools = lib.make_tools(self.workdir_path, bash_allow=bash_allow)

    def initialize_paths(self) -> None:
        os.makedirs(self.paths.run_dir, exist_ok=True)

    def initialize_run_record(self) -> None:
        self.run_toml_data = {
            "run_id": self.run_id,
            "created_at": lib.iso_utc_now(),
            "workdir": str(self.workdir_path),
            "task_file": str(self.task_path),
            "model": str(self.model),
            "agent_loops": int(self.agent_loops),
            "agent_iterations": int(self.agent_iterations),
            "cumulative_input_tokens": 0,
            "cumulative_output_tokens": 0,
            "cumulative_cache_creation_tokens": 0,
            "cumulative_cache_read_tokens": 0,
            "loops_completed": 0,
        }
        lib.atomic_write_toml(self.paths.run_toml, self.run_toml_data)

    def initialize_loop_state(self) -> None:
        self.state = init_state(run_id=self.run_id, workdir=self.workdir_path, task_file=self.task_path)
        write_state(self.paths.state_toml, self.state)

    def update_state(self, *, loop_index: int, phase: str) -> None:
        self.state["run_id"] = self.run_id
        self.state["loop_index"] = loop_index
        self.state["phase"] = phase
        write_state(self.paths.state_toml, self.state)

    def reload_task_context_if_needed(self) -> None:
        try:
            current_mtime = self.task_path.stat().st_mtime
        except OSError:
            current_mtime = 0.0
        if current_mtime != self.task_mtime:
            self.load_task_context()
            self.task_mtime = current_mtime

    def persist_run_progress(self, loop_idx: int) -> None:
        update_cumulative_tokens(self.run_toml_data, self.loop_summaries + self.review_summaries)
        self.run_toml_data["loops_completed"] = loop_idx
        lib.atomic_write_toml(self.paths.run_toml, self.run_toml_data)
        self.loops_completed = loop_idx

    def build_execution_logging(self) -> lib.ToolLogSink:
        exec_log: lib.ToolLogSink = lib.ToolLogToml(path=str(self.paths.execution_toml))
        if self.logging is not None:
            extra_log = lib.ToolLogToml(path=str(self.logging) if str(self.logging) else None)
            exec_log = lib.MultiToolLogSink([exec_log, extra_log])
        return exec_log

    def build_execution_agent(self, logging: lib.ToolLogSink) -> lib.Agent:
        return lib.Agent(
            self.neural_model,
            tools=self.tools,
            max_iterations=self.agent_iterations,
            show_diagnostics=self.verbose,
            logging=logging,
        )

    def read_initial_task_content(self, loop_idx: int) -> str:
        if loop_idx != 1:
            return ""
        try:
            return (self.workdir_path / self.task_rel).read_text(errors="replace")
        except OSError:
            return ""

    def run_execution_phase(self, loop_idx: int) -> tuple[LoopSummary, str, Optional[str]]:
        if self.verbose:
            print(f"\n▶  loop {loop_idx} [execution]")

        self.update_state(loop_index=loop_idx, phase="execution")
        lib.atomic_write_text(self.paths.execution_toml, "")

        exec_log = self.build_execution_logging()
        exec_agent = self.build_execution_agent(exec_log)
        task_content = self.read_initial_task_content(loop_idx)

        exec_task = build_execution_task(
            workdir=self.workdir_path,
            task_rel=self.task_rel,
            task_content=task_content,
            loop_idx=loop_idx,
            agent_loops=self.agent_loops,
            loop_summaries=self.loop_summaries,
            pending_items=self.pending_items,
        )

        exec_result = exec_agent.run(exec_task, system=self.system, chat_history=self.chat_history)
        exec_answer = exec_result.final_text or ""
        loop_summary = loop_summary_from_result(loop_idx, exec_result)

        self.loop_summaries.append(loop_summary)
        self.pending_items = extract_pending_items(exec_answer)
        self.persist_run_progress(loop_idx)
        return loop_summary, exec_answer, extract_status(exec_answer)

    def build_review_tools(self) -> List[lib.AgentTool]:
        review_bash_allow = {"ls", "stat", "pwd", "find", "grep", "head", "tail", "which", "env", "rg"}
        review_tools = [t for t in self.tools if t.name in ("read", "glob", "write")]
        review_tools.append(
            lib.BashTool(cwd=self.workdir_path, bounded=True, allowed_commands=review_bash_allow)
        )
        return review_tools

    def build_review_agent(self) -> lib.Agent:
        return lib.Agent(
            self.neural_model,
            tools=self.build_review_tools(),
            max_iterations=max(6, self.agent_iterations // 2),
            show_diagnostics=self.verbose,
            logging=lib.ToolLogToml(path=str(self.paths.run_dir / "review.toml")),
        )

    def apply_review_data(self, review_data: Dict[str, Any]) -> bool:
        raw_pending = review_data.get("pending") or []
        if isinstance(raw_pending, list):
            reviewed_items = [
                p.get("item", "") if isinstance(p, dict) else str(p)
                for p in raw_pending
                if p
            ]
            self.pending_items = [x for x in reviewed_items if x]

        blocker = review_data.get("blocker", "")
        if not self.pending_items and not blocker:
            return True
        return False

    def run_review_phase(self, loop_idx: int, loop_summary: LoopSummary, exec_answer: str) -> bool:
        if self.verbose:
            print(f"\n▶  loop {loop_idx} [review]")

        self.update_state(loop_index=loop_idx, phase="review")
        review_agent = self.build_review_agent()

        review_task = build_review_task(
            workdir=self.workdir_path,
            task_rel=self.task_rel,
            review_output_rel=self.review_output_rel,
            loop_index=loop_idx,
            loop_summary=loop_summary,
            exec_answer=exec_answer or "",
        )

        review_result = review_agent.run(review_task, system=self.system)
        review_loop_summary = loop_summary_from_result(loop_idx, review_result)
        self.review_summaries.append(review_loop_summary)
        self.persist_run_progress(loop_idx)

        review_data = read_review_output(self.paths.review_output_toml)
        if not review_data:
            return False

        if self.apply_review_data(review_data):
            if self.verbose:
                print(
                    f"\n✓ No pending items and no blocker after loop {loop_idx} "
                    "— task complete, stopping early."
                )
            return True

        return False

    def run(self) -> str:
        for loop_idx in range(1, self.agent_loops + 1):
            self.reload_task_context_if_needed()
            loop_summary, exec_answer, status = self.run_execution_phase(loop_idx)

            if status == "COMPLETE":
                if self.verbose:
                    print(
                        f"\n✓ execution agent reported STATUS: COMPLETE after loop {loop_idx} "
                        "— task complete, stopping early."
                    )
                self.pending_items = []
                break

            if self.run_review_phase(loop_idx, loop_summary, exec_answer):
                break

        return (
            f"completed {self.loops_completed}/{self.agent_loops} loop(s) "
            f"— run: {self.run_id} "
            f"— artifacts: {self.paths.run_dir}"
        )


def loop_summary_from_result(loop_index: int, result: "lib.RunResult") -> LoopSummary:
    """Build a LoopSummary directly from an Agent RunResult.

    This consumes the structured result the agent returns rather than re-parsing
    the TOML event log, so the loop no longer couples to the event-log schema.
    """

    summary = LoopSummary(loop_index=loop_index)

    for tr in result.tool_results:
        tool = tr.name or "?"
        args = tr.args if isinstance(tr.args, dict) else {}
        if tool == "bash":
            ap = (args.get("command") or "").replace("\n", " ").strip()[:80]
        elif tool in ("read", "write", "edit"):
            ap = str(args.get("path", ""))
        else:
            ap = ""

        preview = (tr.output_preview or "").strip()

        if not tr.ok:
            err_msg = preview[:80] if preview else "unknown error"
            summary.errors.append(f"{tool}({ap}): {err_msg}")
        elif tool == "write":
            summary.files_written.append(ap)
        elif tool == "edit":
            summary.files_edited.append(ap)
        elif tool == "read":
            summary.files_read.append(ap)
        elif tool == "bash":
            first = (preview.splitlines() or [""])[0][:60]
            summary.commands_run.append(f"{ap}  →  {first}")

    # Calls the agent attempted but the harness refused — never reached the
    # workspace, so they are tracked separately from verified actions/errors.
    for rc in result.rejected_calls:
        tool = rc.name or "?"
        args = rc.args if isinstance(rc.args, dict) else {}
        if tool == "bash":
            ap = (args.get("command") or "").replace("\n", " ").strip()[:80]
        elif tool in ("read", "write", "edit"):
            ap = str(args.get("path", ""))
        else:
            ap = ""
        summary.rejected.append(f"{tool}({ap}): {rc.reason}")

    u = result.usage
    summary.total_input_tokens = u.input
    summary.total_output_tokens = u.output
    summary.total_cache_creation_tokens = u.cache_write
    summary.total_cache_read_tokens = u.cache_read

    return summary


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_system_prompt(*, workdir: Path) -> str:
    return (
        "You are an autonomous coding agent specializing in test-driven development and repair.\n\n"
        "Core rules:\n"
        "- NEVER fabricate command results, test output, or file contents — always call a tool.\n"
        "- Determine project state by reading files and running commands; never assume or infer.\n"
        "- Only access files and paths within the working directory; never target system directories.\n"
        "- In bash commands, only use relative paths or paths under the working directory.\n\n"
        "Efficiency rules (critical for speed):\n"
        "- Batch ALL independent reads and globs into the first turn — fetch everything you need before acting.\n"
        "- Once you have identified the changes needed, implement them immediately without further exploration.\n"
        "- Do not re-read files you have already read unless they were modified since your last read.\n"
        "- Prefer a single comprehensive edit over multiple small edits to the same file.\n"
        "- Run tests after each meaningful code change — do not defer validation to the end.\n"
        "- When all tests pass and all pending items are resolved, stop immediately — do not invent further work.\n\n"
        "Tool guidance:\n"
        "- Use glob to discover structure, read for content, bash for running commands and tests.\n"
        "- Use edit for targeted changes; use write only when creating new files or doing full rewrites.\n"
    )


def format_loop_context(
    *,
    loop_idx: int,
    agent_loops: int,
    loop_summaries: List[LoopSummary],
    pending_items: List[str],
) -> str:
    """Build the injected context block that orients the execution agent.

    This replaces the agent having to read state/history/plan files.
    """
    lines = [f"Loop {loop_idx} of {agent_loops}."]

    # Accumulate the complete file inventory across ALL prior loops so the
    # agent knows what exists without re-globbing or re-reading orientation files.
    all_written: List[str] = []
    all_edited: List[str] = []
    seen_w: set = set()
    seen_e: set = set()
    for s in loop_summaries:
        for f in s.files_written:
            if f not in seen_w:
                seen_w.add(f)
                all_written.append(f)
        for f in s.files_edited:
            if f not in seen_e:
                seen_e.add(f)
                all_edited.append(f)

    if all_written:
        lines.append("Files created across all prior loops: " + ", ".join(all_written))
    if all_edited:
        lines.append("Files edited across all prior loops: " + ", ".join(all_edited))

    # Show last loop's commands/errors for immediate context.
    if loop_summaries:
        last = loop_summaries[-1]
        parts: List[str] = []
        if last.commands_run:
            parts.append("ran: " + "; ".join(last.commands_run[:3]))
        if last.errors:
            parts.append("errors: " + "; ".join(last.errors[:3]))
        if parts:
            lines.append("Last loop — " + " | ".join(parts) + ".")

    if pending_items:
        lines.append("Pending next steps:")
        for item in pending_items[:5]:
            lines.append(f"  - {item}")
    else:
        lines.append("No pending steps recorded — determine next action from the task file.")

    return "\n".join(lines)


def build_execution_task(
    *,
    workdir: Path,
    task_rel: str,
    task_content: str = "",
    loop_idx: int,
    agent_loops: int,
    loop_summaries: List[LoopSummary],
    pending_items: List[str],
) -> str:
    context = format_loop_context(
        loop_idx=loop_idx,
        agent_loops=agent_loops,
        loop_summaries=loop_summaries,
        pending_items=pending_items,
    )
    if loop_idx == 1 and task_content:
        # Pre-inject the spec so the agent skips the read-to-orient round-trip.
        orient_step = (
            "1. The full task specification is provided below — do NOT re-read the task file.\n\n"
            f"<task_specification>\n{task_content}\n</task_specification>\n"
        )
    elif loop_idx == 1:
        orient_step = "1. Read the task file to orient yourself on the specification.\n"
    else:
        orient_step = (
            "1. The specification and all prior work are already summarised above — "
            "do NOT re-read the task file or re-glob the workspace to orient yourself.\n"
        )
    return (
        f"{context}\n\n"
        f"Working directory: {workdir}\n"
        f"Task file: {task_rel} (read-only — contains the full specification)\n\n"
        "PHASE: EXECUTION\n\n"
        "Goal: COMPLETE the task in this single session if at all possible. Implement every\n"
        "remaining item you can, verify it, and only stop when the task is fully done or you\n"
        "are genuinely blocked. Do NOT implement just one item and defer the rest to a later\n"
        "loop — keep working until everything that can be done this session is done.\n\n"
        "Protocol:\n"
        + orient_step +
        "2. Write a short PLAN (3–7 bullets) covering everything you intend to complete now.\n"
        "3. Execute the plan autonomously and to completion — do NOT ask for confirmation, and\n"
        "   do NOT stop after a single change while more work remains.\n"
        "4. Before each set of tool calls, write one or two sentences stating what you are about to do and why.\n"
        "5. Inspect the current state of relevant files before editing them.\n"
        "6. After each meaningful change, run the closest available check (tests, lint, typecheck).\n"
        "   If none exists, run `python -m compileall .`. Do not defer all validation to the end.\n"
        "7. Continue until every task is implemented AND the checks pass, or you hit a hard blocker.\n\n"
        "Finish with <final_answer> that includes, in this order:\n"
        "- A status line of the EXACT form `STATUS: COMPLETE` if every task is implemented and all\n"
        "  checks pass, otherwise `STATUS: INCOMPLETE`.\n"
        "- The PLAN you followed (final form).\n"
        "- Exact output of the checks you ran (verbatim, not paraphrased).\n"
        "- ONLY when STATUS is INCOMPLETE: a bulleted list of NEXT STEPS (3-5 concrete items) for\n"
        "  the following loop, beginning with the exact line: NEXT STEPS:\n"
    )


def build_review_task(
    *,
    workdir: Path,
    task_rel: str,
    review_output_rel: str,
    loop_index: int,
    loop_summary: LoopSummary,
    exec_answer: str = "",
) -> str:
    # Build a text representation of the harness-computed summary so the
    # review agent has verified facts without needing to parse any transcript.
    summary_lines = [f"## Verified actions from loop {loop_index} (harness-computed)"]
    if loop_summary.files_read:
        summary_lines.append("Files read: " + ", ".join(loop_summary.files_read))
    if loop_summary.files_written:
        summary_lines.append("Files written: " + ", ".join(loop_summary.files_written))
    if loop_summary.files_edited:
        summary_lines.append("Files edited: " + ", ".join(loop_summary.files_edited))
    if loop_summary.commands_run:
        summary_lines.append("Commands run:")
        for c in loop_summary.commands_run:
            summary_lines.append(f"  {c}")
    if loop_summary.errors:
        summary_lines.append("Errors:")
        for e in loop_summary.errors:
            summary_lines.append(f"  {e}")
    if not (loop_summary.files_read or loop_summary.files_written or loop_summary.files_edited
            or loop_summary.commands_run or loop_summary.errors):
        summary_lines.append("  (no verified actions)")
    verified_block = "\n".join(summary_lines)

    # Calls the agent attempted but the harness refused to run. These did NOT
    # touch the workspace, so any execution-report claim that depends on one of
    # them is unverified and must be flagged.
    rejected_block = ""
    if loop_summary.rejected:
        rejected_lines = [
            "## Attempted but NOT executed (harness-rejected — no workspace effect)"
        ]
        for r in loop_summary.rejected:
            rejected_lines.append(f"  {r}")
        rejected_block = "\n".join(rejected_lines)

    return (
        f"Working directory: {workdir}\n"
        f"Task file: {task_rel} (read-only)\n"
        f"Review output: {review_output_rel} (write here)\n\n"
        "PHASE: REVIEW\n\n"
        f"{verified_block}\n\n"
        + (f"{rejected_block}\n\n" if rejected_block else "")
        + (
            f"\n[Execution agent report — loop {loop_index}]\n\n{exec_answer}\n\n"
            if exec_answer else ""
        )
        + "Your job:\n"
        "1. Cross-reference the execution report's claims against the verified actions above.\n"
        "   File reads listed above are verified — treat the agent's claims about those\n"
        "   files as trustworthy. Only flag claims about files NOT in the verified list.\n"
        "   Any claim that relies on an 'Attempted but NOT executed' call did NOT actually\n"
        "   happen — flag it as unverified and carry the underlying work into pending items.\n"
        "2. Write your assessment to the review output file in TOML format:\n\n"
        "   ```toml\n"
        f"   loop = {loop_index}\n"
        "   summary = \"One paragraph describing what actually happened.\"\n"
        "   blocker = \"Main current blocker, or empty string if none.\"\n\n"
        "   [[pending]]\n"
        "   item = \"First concrete next step\"\n"
        "   ```\n\n"
        "Rules:\n"
        "- If tests passed and no errors are listed above, set blocker = \"\" and leave pending empty.\n"
        "- pending items must be concrete and actionable (not 'continue working').\n"
        "- Limit to 5 pending items maximum.\n\n"
        "Finish with <final_answer> confirming you wrote the review output file.\n"
    )


# ---------------------------------------------------------------------------
# Runner utilities
# ---------------------------------------------------------------------------

def ensure_within_workdir(path: Path, workdir: Path) -> Path:
    path = path.resolve()
    workdir = workdir.resolve()
    try:
        path.relative_to(workdir)
    except ValueError:
        raise ValueError(f"Path {path} is outside working directory {workdir}")
    return path


def extract_pending_items(final_text: str) -> List[str]:
    """Parse NEXT STEPS: section from an execution agent's final_answer."""
    items: List[str] = []
    in_section = False
    for line in (final_text or "").splitlines():
        if line.strip().upper().startswith("NEXT STEPS"):
            in_section = True
            continue
        if in_section:
            stripped = line.strip()
            if not stripped:
                continue
            # Stop at the next markdown heading
            if stripped.startswith("#"):
                break
            if stripped.startswith("-") or stripped.startswith("*"):
                items.append(stripped.lstrip("-* ").strip())
            elif stripped[0].isdigit() and stripped[1:3] in (". ", ") "):
                items.append(stripped[2:].strip() if len(stripped) > 2 else stripped)
    return items[:5]


def extract_status(final_text: str) -> Optional[str]:
    """Parse a `STATUS: COMPLETE` / `STATUS: INCOMPLETE` line from a final answer.

    Returns "COMPLETE", "INCOMPLETE", or None if no status line was emitted.
    """
    for line in (final_text or "").splitlines():
        s = line.strip().upper()
        if s.startswith("STATUS:"):
            val = s[len("STATUS:"):].strip()
            if val.startswith("COMPLETE"):
                return "COMPLETE"
            if val.startswith("INCOMPLETE"):
                return "INCOMPLETE"
    return None


def update_cumulative_tokens(
    run_toml_data: Dict[str, Any], summaries: List[LoopSummary]
) -> None:
    """Roll up per-loop token totals into the run.toml record (in place)."""
    run_toml_data["cumulative_input_tokens"] = sum(s.total_input_tokens for s in summaries)
    run_toml_data["cumulative_output_tokens"] = sum(s.total_output_tokens for s in summaries)
    run_toml_data["cumulative_cache_creation_tokens"] = sum(
        s.total_cache_creation_tokens for s in summaries
    )
    run_toml_data["cumulative_cache_read_tokens"] = sum(
        s.total_cache_read_tokens for s in summaries
    )


def read_review_output(path: Path) -> Dict[str, Any]:
    """Read the review agent's structured TOML output."""
    data = lib.read_toml(path)
    return data


def prompt_loop(
    task_file: Union[Path, str],
    model: Union[Path, str],
    agent_loops: int = 5,
    agent_iterations: int = 30,
    verbose: bool = False,
    logging: Optional[Union[Path, str]] = None,
    workdir: Optional[Union[Path, str]] = None,
    reason: bool = False,
) -> str:
    """Run a bounded execution → review loop.

    Cross-loop state is carried entirely in-memory (loop_summaries, pending_items).
    Agents receive context injected into their task strings — they do not read
    state or history files to orient themselves.

    Per loop:
      1) EXECUTION agent: reads task file + does work + reports STATUS + NEXT STEPS.
         If it reports STATUS: COMPLETE, the loop exits early (review skipped).
      2) REVIEW agent: receives harness-computed action summary + writes review_output.toml.

    Persistent files under `.codescribe/loop/` (for inspection / crash-resume):
      run.toml             — run configuration
      state.toml           — current loop index + phase
      execution.toml       — raw TOML event log for the most recent execution phase
      review_output.toml   — review agent's structured output
    """
    runner = PromptLoopRunner(
        task_file=task_file,
        model=model,
        agent_loops=agent_loops,
        agent_iterations=agent_iterations,
        verbose=verbose,
        logging=logging,
        workdir=workdir,
        reason=reason,
    )
    return runner.run()
