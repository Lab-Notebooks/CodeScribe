# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Bounded author → review loop (`code-scribe loop`).

Each loop runs an AUTHOR agent, then the harness records the workspace diff
and runs the task's [check]. A REVIEW agent then judges that evidence and
writes review_output.toml (open items, blocker, rules). Cross-loop state lives
in memory and is injected into the next task prompt; files under
`.codescribe/loop/` are for inspection only. Convergence is decided by the
harness: see docs/loop.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from codescribe.lib.agent import Agent, RunResult
from codescribe.lib.util import (
    MultiToolLogSink,
    Timer,
    ToolLogSink,
    ToolLogToml,
    atomic_write_text,
    atomic_write_toml,
    ensure_loop_metadata_dir,
    iso_utc_now,
    load_chat_template,
    new_run_id,
    read_toml,
    write_loop_manifest,
    write_loop_phase_metadata,
)
from codescribe.lib.models import set_neural_model
from codescribe.lib.tools import (
    AgentTool,
    BashTool,
    ensure_within_workdir,
    make_tools,
    resolve_script_dir,
)
from codescribe.lib.loop._check import (
    CheckResult,
    run_check,
    workspace_diff,
    workspace_tree,
)
from codescribe.lib.loop._guard import GuardedBashTool, ProtectedFileGuard
from codescribe.lib.loop._prompts import (
    AUTHOR_SYSTEM,
    REVIEW_SYSTEM,
    build_author_task,
    build_review_task,
    extract_pending_items,
    extract_status,
)
from codescribe.lib.loop._state import (
    MAX_ITEMS,
    MAX_REVIEW_RULES,
    LoopRule,
    LoopScore,
    LoopSummary,
    ReviewItem,
    get_loop_paths,
    init_state,
    parse_rules,
    protect_conflict,
    update_cumulative_tokens,
    write_state,
)

REVIEW_BASH = {
    "ls",
    "stat",
    "pwd",
    "find",
    "grep",
    "head",
    "tail",
    "which",
    "env",
    "rg",
    "git",
}


def _reasoning_config(model: Any) -> str:
    thinking = getattr(model, "thinking", None)
    if thinking:
        return "thinking=" + ",".join(f"{k}:{v}" for k, v in sorted(thinking.items()))
    effort = getattr(model, "reasoning_effort", None)
    return f"reasoning_effort={effort}" if effort else "provider-default"


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
    stall_loops: int = 2

    def __post_init__(self) -> None:
        self.workdir_path = Path(self.workdir or Path.cwd()).resolve()
        self.task_path = ensure_within_workdir(Path(self.task_file), self.workdir_path)
        self.task_rel = str(self.task_path.relative_to(self.workdir_path))
        self.neural_model = set_neural_model(self.model, reasoning=self.reason)
        self.run_id = new_run_id()
        self.paths = get_loop_paths(self.workdir_path)
        self.review_output_rel = str(
            self.paths.review_output_toml.relative_to(self.workdir_path)
        )

        self.loop_summaries: List[LoopSummary] = []
        self.review_summaries: List[LoopSummary] = []
        self.scores: List[LoopScore] = []
        self.items: List[ReviewItem] = []
        self.next_item_id = 1
        self.blocker = ""
        self.author_next_steps: List[str] = []
        self.rules: List[LoopRule] = []
        self.rule_problems: List[str] = []
        self.new_rule_problems: List[str] = []  # from the latest review only
        self.guard = ProtectedFileGuard(self.workdir_path)
        self.stop_reason = "max_loops"
        self.loops_completed = 0
        self.task_mtime = 0.0
        self.phase_metadata_files: List[Path] = []

        self.load_task_context()
        os.makedirs(self.paths.run_dir, exist_ok=True)
        ensure_loop_metadata_dir(self.paths.run_dir)
        self.write_rules()
        self.run_doc = {
            "run_id": self.run_id,
            "created_at": iso_utc_now(),
            "workdir": str(self.workdir_path),
            "task_file": str(self.task_path),
            "model": str(self.model),
            "agent_loops": int(self.agent_loops),
            "agent_iterations": int(self.agent_iterations),
            "reason": bool(self.reason),
            "max_tokens": int(getattr(self.neural_model, "max_tokens", 0)),
            "reasoning_config": _reasoning_config(self.neural_model),
        }
        update_cumulative_tokens(self.run_doc, [])
        self.run_doc["loops_completed"] = 0
        atomic_write_toml(self.paths.run_toml, self.run_doc)
        self.state = init_state(
            run_id=self.run_id, workdir=self.workdir_path, task_file=self.task_path
        )
        write_state(self.paths.state_toml, self.state)

    # -- task file -----------------------------------------------------------

    def load_task_context(self) -> None:
        """Read the task file once. Tool policy is never re-read mid-run: the
        agent could otherwise grant itself commands by editing the file."""
        self.chat_history, meta = load_chat_template(self.task_path, return_meta=True)
        tools_cfg = meta.get("tools") or {}
        self.bash_allow = set(tools_cfg.get("bash") or [])
        self.memory = tools_cfg.get("memory")
        self.check = meta.get("check")
        self.writable = list(tools_cfg.get("writable") or [])
        self.python_scripts_dir()  # fail fast if memory escapes the workdir
        self.guard.add(self.task_rel)  # the task file is always protected
        for pattern in tools_cfg.get("protect") or []:
            self.guard.add(pattern)
        self.base_protect = list(self.guard.patterns)

    def reload_task_content(self) -> None:
        """Refresh only the chat history when the task file changed."""
        try:
            mtime = self.task_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        if mtime != self.task_mtime:
            self.chat_history = load_chat_template(self.task_path)
            self.task_mtime = mtime

    def bash_allowed(self) -> List[str]:
        return sorted(BashTool.DEFAULT_ALLOWED | self.bash_allow)

    def python_scripts_dir(self) -> str:
        """Workdir-relative dir allowlisted python must run from ("" if no python)."""
        path = resolve_script_dir(
            self.workdir_path, set(self.bash_allowed()), self.memory
        )
        return str(path.relative_to(self.workdir_path)) if path else ""

    def protected_paths(self) -> set:
        return set(self.guard.resolve()) | {self.task_path.resolve()}

    # -- persistence ---------------------------------------------------------

    def update_state(self, loop_index: int, phase: str) -> None:
        self.state.update(run_id=self.run_id, loop_index=loop_index, phase=phase)
        write_state(self.paths.state_toml, self.state)

    def write_rules(self) -> None:
        atomic_write_toml(
            self.paths.rules_toml,
            {
                "rule": [r.to_toml() for r in self.rules],
                "protected": self.guard.relative(),
                "problems": list(self.rule_problems),
            },
        )

    def persist_run_progress(self, loop_idx: int) -> None:
        update_cumulative_tokens(
            self.run_doc, self.loop_summaries + self.review_summaries
        )
        self.run_doc["loops_completed"] = loop_idx
        self.run_doc["stop_reason"] = self.stop_reason
        self.run_doc["score"] = [s.to_toml() for s in self.scores]
        atomic_write_toml(self.paths.run_toml, self.run_doc)
        write_loop_manifest(
            self.paths.metadata_dir,
            run_doc=self.run_doc,
            phase_files=self.phase_metadata_files,
        )
        self.loops_completed = loop_idx

    def record_phase(
        self, loop_idx: int, phase: str, result: RunResult, duration_s: float
    ) -> None:
        self.phase_metadata_files.append(
            write_loop_phase_metadata(
                self.paths.metadata_dir,
                run_id=self.run_id,
                loop_index=loop_idx,
                phase=phase,
                model=str(self.model),
                task_file=str(self.task_path),
                workdir=str(self.workdir_path),
                stop_reason=result.stop_reason,
                final_text_present=bool(result.final_text),
                usage=result.usage,
                iterations=result.iterations,
                tool_results=result.tool_results,
                rejected_calls=result.rejected_calls,
                duration_s=duration_s,
            )
        )
        self.persist_run_progress(loop_idx)

    # -- author phase --------------------------------------------------------

    def build_author_agent(self) -> Agent:
        # Tools are rebuilt every phase so reviewer-added protections apply.
        tools = make_tools(
            self.workdir_path,
            bash_allow=self.bash_allow,
            protected_paths=self.protected_paths(),
            memory=self.memory,
        )
        tools = [
            GuardedBashTool(t, rules=self.rules, guard=self.guard)
            if t.name == "bash"
            else t
            for t in tools
        ]
        log: ToolLogSink = ToolLogToml(path=str(self.paths.author_toml))
        if self.logging is not None:
            log = MultiToolLogSink([log, ToolLogToml(path=str(self.logging) or None)])
        return Agent(
            self.neural_model,
            tools=tools,
            max_iterations=self.agent_iterations,
            show_diagnostics=self.verbose,
            logging=log,
        )

    def run_check(self) -> Optional[CheckResult]:
        if not self.check:
            return None
        result = run_check(
            self.check["command"], cwd=self.workdir_path, timeout=self.check["timeout"]
        )
        if self.verbose:
            print(f"  check `{result.command}`: {'PASS' if result.passed else 'FAIL'}")
        return result

    def run_author_phase(self, loop_idx: int) -> Tuple[LoopSummary, str, Optional[str]]:
        if self.verbose:
            print(f"\n▶  loop {loop_idx} [author]")
        self.update_state(loop_idx, "author")
        atomic_write_text(self.paths.author_toml, "")
        agent = self.build_author_agent()
        task = build_author_task(
            workdir=self.workdir_path,
            task_rel=self.task_rel,
            loop_idx=loop_idx,
            agent_loops=self.agent_loops,
            loop_summaries=self.loop_summaries,
            items=self.items,
            blocker=self.blocker,
            rules=self.rules,
            protected=self.guard.relative(),
            python_scripts_dir=self.python_scripts_dir(),
            bash_allowed=self.bash_allowed(),
        )

        self.guard.snapshot()
        tree_before = workspace_tree(self.workdir_path)
        timer = Timer()
        result = agent.run(task, system=AUTHOR_SYSTEM, chat_history=self.chat_history)
        duration_s = timer.ms / 1000.0
        self.guard.restore_changes()  # catch-all for changes made outside bash
        tree_after = workspace_tree(self.workdir_path)

        answer = result.final_text or ""
        summary = LoopSummary.from_result(loop_idx, result)
        summary.violations = list(self.guard.violations)
        summary.diff = workspace_diff(self.workdir_path, tree_before, tree_after)
        if tree_before:
            summary.changed = tree_before != tree_after
        else:
            summary.changed = bool(summary.files_written or summary.files_edited)
        summary.check = self.run_check()
        if self.verbose and summary.violations:
            print("  ⚠ protected files reverted: " + "; ".join(summary.violations))

        self.loop_summaries.append(summary)
        self.author_next_steps = extract_pending_items(answer)
        self.record_phase(loop_idx, "author", result, duration_s)
        return summary, answer, extract_status(answer)

    # -- review phase --------------------------------------------------------

    def build_review_agent(self) -> Agent:
        protected = self.protected_paths()
        tools: List[AgentTool] = [
            t
            for t in make_tools(self.workdir_path, protected_paths=protected)
            if t.name in ("read", "glob", "write")
        ]
        tools.append(
            BashTool(
                cwd=self.workdir_path,
                bounded=True,
                allowed_commands=REVIEW_BASH,
                protected_paths=protected,
            )
        )
        return Agent(
            self.neural_model,
            tools=tools,
            max_iterations=max(6, self.agent_iterations // 2),
            show_diagnostics=self.verbose,
            logging=ToolLogToml(path=str(self.paths.run_dir / "review.toml")),
        )

    def run_review_phase(
        self, loop_idx: int, summary: LoopSummary, author_answer: str
    ) -> Optional[int]:
        """Run the reviewer; return the resolved-item count, or None if it wrote no output."""
        if self.verbose:
            print(f"\n▶  loop {loop_idx} [review]")
        self.update_state(loop_idx, "review")
        self.paths.review_output_toml.unlink(missing_ok=True)
        agent = self.build_review_agent()
        task = build_review_task(
            workdir=self.workdir_path,
            task_rel=self.task_rel,
            review_output_rel=self.review_output_rel,
            loop_index=loop_idx,
            loop_summary=summary,
            exec_answer=author_answer or "",
            items=self.items,
            rules=self.rules,
            protected=self.guard.relative(),
            rule_problems=self.new_rule_problems,
        )

        timer = Timer()
        result = agent.run(task, system=REVIEW_SYSTEM, chat_history=self.chat_history)
        self.review_summaries.append(LoopSummary.from_result(loop_idx, result))
        self.record_phase(loop_idx, "review", result, timer.ms / 1000.0)

        review = read_toml(self.paths.review_output_toml)
        if not review:
            # No review written: fall back to the author's own next steps.
            self.blocker = ""
            self.update_items(self.author_next_steps, loop_idx)
            return None
        self.blocker = str(review.get("blocker") or "").strip()
        self.apply_review_rules(review)
        return self.update_items(review.get("pending"), loop_idx)

    def update_items(self, raw: Any, loop_idx: int) -> int:
        """Replace open items with the reviewer's list; return how many were resolved."""
        still_open = {item.id: item for item in self.items}
        items: List[ReviewItem] = []
        for entry in raw if isinstance(raw, list) else []:
            if isinstance(entry, str):
                entry = {"item": entry}
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("item") or "").strip()
            if not text:
                continue
            prior = still_open.pop(str(entry.get("id") or "").strip(), None)
            if prior:
                items.append(ReviewItem(prior.id, text, prior.opened))
            else:
                items.append(ReviewItem(f"P{self.next_item_id}", text, loop_idx))
                self.next_item_id += 1
        self.items = items[:MAX_ITEMS]
        return len(still_open)

    def apply_review_rules(self, review: Dict[str, Any]) -> None:
        """Retire and add reviewer rules, then rebuild the protected set."""
        retire = review.get("retire") or []
        if isinstance(retire, list):
            retire_ids = {str(x).strip() for x in retire}
            self.rules[:] = [r for r in self.rules if r.id not in retire_ids]

        next_id = (
            max((int(r.id[1:]) for r in self.rules if r.id[1:].isdigit()), default=0)
            + 1
        )
        new_rules, problems = parse_rules(
            review.get("rule"),
            source=f"review:{review.get('loop', '?')}",
            start_id=next_id,
        )
        # A rule may not lock a file the author must change: drop the protect, keep the text.
        for rule in new_rules:
            hit = rule.protect and protect_conflict(
                rule.protect, self.writable, self.workdir_path
            )
            if hit:
                problems.append(
                    f"rule protect {rule.protect!r} targets author-writable {hit!r}; ignored"
                )
                rule.protect = ""
        known = {r.text for r in self.rules}
        self.rules.extend(r for r in new_rules if r.text not in known)
        del self.rules[: max(0, len(self.rules) - MAX_REVIEW_RULES)]
        self.rule_problems.extend(problems)
        self.new_rule_problems = problems

        # Base protections last the whole run; a rule's protect lasts while the rule does.
        self.guard.patterns = list(self.base_protect)
        for rule in self.rules:
            self.guard.add(rule.protect)
        self.write_rules()

    # -- convergence ---------------------------------------------------------

    def record_score(self, summary: LoopSummary, resolved: int) -> LoopScore:
        """Progress = the workspace changed and an item was resolved or the check turned green."""
        prev = self.scores[-1] if self.scores else None
        check_passed = summary.check.passed if summary.check else None
        check_fixed = bool(check_passed) and not (prev and prev.check_passed)
        score = LoopScore(
            loop_index=summary.loop_index,
            changed=summary.changed,
            check_passed=check_passed,
            resolved=resolved,
            open_items=len(self.items),
            progress=summary.changed and (prev is None or resolved > 0 or check_fixed),
        )
        self.scores.append(score)
        return score

    def stalled(self) -> bool:
        recent = self.scores[-self.stall_loops :]
        return len(recent) == self.stall_loops and not any(s.progress for s in recent)

    def stop(self, reason: str, message: str) -> None:
        self.stop_reason = reason
        if self.verbose:
            print(f"\n■ {message}")

    def run(self) -> str:
        for loop_idx in range(1, self.agent_loops + 1):
            self.reload_task_content()
            summary, author_answer, status = self.run_author_phase(loop_idx)
            check_ok = summary.check is None or summary.check.passed

            # Self-reported completion counts only with a passing check.
            if status == "COMPLETE" and summary.check and summary.check.passed:
                self.items, self.blocker = [], ""
                self.stop(
                    "complete", f"author COMPLETE and check passed (loop {loop_idx})"
                )
                break

            resolved = self.run_review_phase(loop_idx, summary, author_answer)
            self.record_score(summary, resolved or 0)
            self.persist_run_progress(loop_idx)

            if (
                resolved is not None
                and check_ok
                and not self.items
                and not self.blocker
            ):
                self.stop(
                    "complete", f"review clean and check passed (loop {loop_idx})"
                )
                break
            if self.stalled():
                self.stop(
                    "stalled",
                    f"no progress for {self.stall_loops} loops (loop {loop_idx})",
                )
                break

        self.persist_run_progress(self.loops_completed)
        return (
            f"{self.stop_reason}: {self.loops_completed}/{self.agent_loops} loop(s) "
            f"— run: {self.run_id} — artifacts: {self.paths.run_dir}"
        )


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
    """Run the bounded author → review loop; returns a one-line outcome."""
    return PromptLoopRunner(
        task_file=task_file,
        model=model,
        agent_loops=agent_loops,
        agent_iterations=agent_iterations,
        verbose=verbose,
        logging=logging,
        workdir=workdir,
        reason=reason,
    ).run()
