# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""_loop

Prompt-loop implementation with in-memory cross-loop state.

State relay design:
  - Within a loop (iteration → iteration): message history + WORKSPACE CONTEXT block.
  - Across loops (loop N → loop N+1): Python objects held in prompt_loop().
    The harness builds LoopSummary from the agent's structured RunResult after
    each author phase and injects it directly into the next loop's task string.
    Agents never read state files to orient themselves.

Persistent files (for inspection / crash-resume only):
  run.toml        — run configuration
  state.toml      — current loop index + phase
  author.toml     — raw event log for the most recent author phase
  review_output.toml — structured output from the review agent (pending items)
  rules.toml      — active reviewer-added rules, for inspection
  metadata/       — durable normalized per-phase telemetry for plotting

Policy (model-agnostic, enforced by the harness rather than by prompt compliance):
  - Protected files: the task file plus [tools].protect. write/edit refuse
    them, and after every author bash call (and at phase end) any change to
    them is reverted and recorded as a violation.
  - Rules: added/retired by the review agent via [[rule]] / retire in
    review_output.toml. Every rule is injected into the
    author task; a rule with `deny_bash` also makes the harness refuse matching
    author bash commands, and a rule with `protect` adds a protected file while
    the rule is active. [tools].writable paths (author-owned files such as a
    work log) can never be protected by a rule.

Convergence (decided by the harness, not by agent self-report):
  - Check: the task file's [check] command runs after every author phase; its
    exit code is ground truth for both agents.
  - Feedback: the reviewer sees the phase's workspace diff and check output;
    the next author sees the check output, the blocker and open review items.
  - Items: review items keep stable ids; open items the reviewer omits are resolved.
  - Stop: complete when the check passes and the review is clean; stalled when
    `stall_loops` consecutive loops make no progress.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
import tempfile
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
    "LoopRule",
    "LoopScore",
    "ReviewItem",
    "CheckResult",
    "ProtectedFileGuard",
    "GuardedBashTool",
    "run_check",
    "workspace_tree",
    "workspace_diff",
    "build_system_prompt",
    "build_review_system_prompt",
    "build_author_task",
    "build_review_task",
    "ensure_within_workdir",
    "prompt_loop",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass
class LoopPaths:
    """Paths under `.codescribe/loop/` for a single shared run."""

    run_dir: Path
    metadata_dir: Path
    run_toml: Path
    state_toml: Path
    author_toml: Path  # raw event log (overwritten each author phase)
    review_output_toml: Path  # review agent's structured output (overwritten each review)
    rules_toml: Path  # active policy rules (written for inspection, never read back)


def get_loop_paths(workdir: Path) -> LoopPaths:
    run_dir = workdir / ".codescribe" / "loop"
    return LoopPaths(
        run_dir=run_dir,
        metadata_dir=run_dir / "metadata",
        run_toml=run_dir / "run.toml",
        state_toml=run_dir / "state.toml",
        author_toml=run_dir / "author.toml",
        review_output_toml=run_dir / "review_output.toml",
        rules_toml=run_dir / "rules.toml",
    )


# ---------------------------------------------------------------------------
# State persistence (TOML)
# ---------------------------------------------------------------------------

_VALID_PHASES = frozenset({"idle", "author", "review"})


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
        raise ValueError(
            f"state['phase'] must be one of {sorted(_VALID_PHASES)}, got {phase!r}"
        )
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
class LoopRule:
    """A binding constraint on the author agent, added by the review agent.

    `text` is always injected into the author task. `deny_bash` (regex) makes the
    harness refuse matching author bash commands; `protect` (workdir-relative path
    or glob) adds protected files.
    """

    id: str
    text: str
    deny_bash: str = ""
    protect: str = ""
    source: str = ""  # "review:<loop>"

    def to_toml(self) -> Dict[str, Any]:
        doc: Dict[str, Any] = {"id": self.id, "text": self.text, "source": self.source}
        if self.deny_bash:
            doc["deny_bash"] = self.deny_bash
        if self.protect:
            doc["protect"] = self.protect
        return doc


# Upper bound on active rules; the oldest are dropped first so a
# runaway reviewer cannot grow the author prompt without limit.
MAX_REVIEW_RULES = 12


def parse_rules(
    raw: Any, *, source: str, start_id: int
) -> tuple[List[LoopRule], List[str]]:
    """Parse rule entries (strings or tables) into LoopRules.

    Returns (rules, problems). Invalid or over-broad `deny_bash` patterns are
    dropped (the rule's text is kept) so a bad regex cannot brick the author.
    """
    rules: List[LoopRule] = []
    problems: List[str] = []
    if not isinstance(raw, list):
        return rules, problems
    next_id = start_id
    for entry in raw:
        if isinstance(entry, str):
            entry = {"text": entry}
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        deny = str(entry.get("deny_bash") or "").strip()
        if deny:
            try:
                if re.search(deny, ""):
                    problems.append(f"deny_bash {deny!r} matches every command; ignored")
                    deny = ""
            except re.error as exc:
                problems.append(f"deny_bash {deny!r} is not a valid regex ({exc}); ignored")
                deny = ""
        rules.append(
            LoopRule(
                id=f"R{next_id}",
                text=text,
                deny_bash=deny,
                protect=str(entry.get("protect") or "").strip(),
                source=source,
            )
        )
        next_id += 1
    return rules, problems


def protect_conflict(pattern: str, writable: List[str], workdir: Path) -> str:
    """Return the writable entry a reviewer `protect` pattern would lock, or ""."""
    p = Path(pattern)
    if p.is_absolute():
        try:
            pattern = str(p.resolve().relative_to(workdir.resolve()))
        except ValueError:
            return ""
    pattern = os.path.normpath(pattern)
    hits = {pattern}
    if any(ch in pattern for ch in "*?["):
        hits |= {str(x.relative_to(workdir)) for x in workdir.glob(pattern)}
    for w in writable:
        w = os.path.normpath(w)
        if fnmatch.fnmatch(w, pattern) or any(fnmatch.fnmatch(h, w) for h in hits):
            return w
    return ""


class ProtectedFileGuard:
    """Snapshot protected files and revert any change to them.

    Tool-level path checks cannot see what an allowlisted interpreter (e.g.
    `python3 script.py`) does, so the guard compares file contents after each
    author bash call and at phase end. This is tamper detection and repair,
    not an OS sandbox.
    """

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.patterns: List[str] = []
        self.saved: Dict[Path, Optional[bytes]] = {}
        self.violations: List[str] = []

    def add(self, pattern: str) -> None:
        if pattern and pattern not in self.patterns:
            self.patterns.append(pattern)

    def resolve(self) -> List[Path]:
        paths: List[Path] = []
        for pattern in self.patterns:
            if any(ch in pattern for ch in "*?["):
                candidates = list(self.workdir.glob(pattern))
            else:
                candidates = [self.workdir / pattern]
            for p in candidates:
                try:
                    p = ensure_within_workdir(p, self.workdir)
                except ValueError:
                    continue
                if p not in paths:
                    paths.append(p)
        return paths

    @staticmethod
    def read_bytes(path: Path) -> Optional[bytes]:
        try:
            return path.read_bytes() if path.is_file() else None
        except OSError:
            return None

    def snapshot(self) -> None:
        self.saved = {p: self.read_bytes(p) for p in self.resolve()}
        self.violations = []

    def restore_changes(self) -> List[str]:
        """Revert protected files to the snapshot; return what was reverted."""
        reverted: List[str] = []
        for path, before in self.saved.items():
            after = self.read_bytes(path)
            if after == before:
                continue
            rel = str(path.relative_to(self.workdir))
            try:
                if before is None:
                    path.unlink()
                    reverted.append(f"{rel} (created; removed)")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(before)
                    reverted.append(
                        f"{rel} ({'deleted' if after is None else 'modified'}; restored)"
                    )
            except OSError as exc:
                reverted.append(f"{rel} (changed; restore FAILED: {exc})")
        self.violations.extend(reverted)
        return reverted


class GuardedBashTool(lib.AgentTool):
    """Author bash tool wrapper that enforces deny rules and protected files."""

    def __init__(
        self, inner: lib.AgentTool, *, rules: List[LoopRule], guard: ProtectedFileGuard
    ) -> None:
        super().__init__(
            name=inner.name,
            description=inner.description,
            parameters=inner.parameters,
            enabled=inner.enabled,
        )
        self.inner = inner
        self.rules = rules
        self.guard = guard

    def run(self, args: Dict[str, Any]) -> str:
        command = str(args.get("command") or "")
        for rule in self.rules:
            if rule.deny_bash and re.search(rule.deny_bash, command):
                return f"Error: command refused by rule {rule.id}: {rule.text}"
        output = self.inner.run(args)
        reverted = self.guard.restore_changes()
        if reverted:
            return (
                "Error: this command changed protected file(s); the harness reverted "
                "them and recorded a violation: "
                + "; ".join(reverted)
                + "\n\n"
                + output
            )
        return output


# ---------------------------------------------------------------------------
# Convergence signals
# ---------------------------------------------------------------------------

CHECK_TAIL_CHARS = 4000
MAX_DIFF_CHARS = 20000
MAX_ITEMS = 8


@dataclass
class CheckResult:
    """Outcome of the task file's [check] command."""

    command: str
    passed: bool
    exit_code: int
    tail: str


def run_check(command: str, *, cwd: Path, timeout: int) -> CheckResult:
    """Run the check as argv (no shell) and keep the tail of its output."""
    try:
        proc = subprocess.run(
            shlex.split(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(command, False, -1, f"timed out after {timeout}s")
    except (OSError, ValueError) as exc:
        return CheckResult(command, False, -1, str(exc))
    output = (proc.stdout + proc.stderr).strip()
    return CheckResult(
        command, proc.returncode == 0, proc.returncode, output[-CHECK_TAIL_CHARS:]
    )


def _git(workdir: Path, *args: str, env: Optional[Dict[str, str]] = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    return proc.stdout


def workspace_tree(workdir: Path) -> str:
    """Git tree hash of the workdir (untracked files included), or "" outside git.

    Uses a throwaway index so the user's staging area is never touched.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        try:
            _git(workdir, "add", "-A", "--", ".", ":(exclude).codescribe", env=env)
            return _git(workdir, "write-tree", env=env).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""


def workspace_diff(workdir: Path, before: str, after: str) -> str:
    """Workdir-relative diff between two trees, capped at MAX_DIFF_CHARS."""
    if not before or not after or before == after:
        return ""
    try:
        diff = _git(workdir, "diff", "--relative", before, after)
    except (OSError, subprocess.CalledProcessError):
        return ""
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n[diff truncated; read the files for the rest]\n"
    return diff


@dataclass
class ReviewItem:
    """An open review finding with an id that is stable across loops."""

    id: str
    text: str
    opened: int  # loop that first raised it


@dataclass
class LoopScore:
    """Per-loop convergence metrics."""

    loop_index: int
    changed: bool
    check_passed: Optional[bool]
    resolved: int
    open_items: int
    progress: bool

    def to_toml(self) -> Dict[str, Any]:
        doc = dict(self.__dict__)
        if self.check_passed is None:
            del doc["check_passed"]
        return doc


@dataclass
class LoopSummary:
    """Harness-computed summary of one author phase.

    Built deterministically from the agent's RunResult and the workspace — no LLM involved.
    Carried in-memory by prompt_loop() and injected into subsequent task prompts.
    """

    loop_index: int
    files_written: List[str] = field(default_factory=list)
    files_edited: List[str] = field(default_factory=list)
    files_read: List[str] = field(default_factory=list)
    commands_run: List[str] = field(default_factory=list)  # "cmd → exit_code N"
    errors: List[str] = field(default_factory=list)
    rejected: List[str] = field(
        default_factory=list
    )  # "tool(args): reason" — attempted, never executed
    violations: List[str] = field(
        default_factory=list
    )  # protected-file changes the harness reverted
    changed: bool = False  # the phase changed the workspace
    diff: str = ""
    check: Optional[CheckResult] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_5m_tokens: int = 0
    total_cache_creation_1h_tokens: int = 0


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
        self.workdir_path = (
            Path(self.workdir).resolve() if self.workdir else Path.cwd().resolve()
        )
        self.task_path = ensure_within_workdir(Path(self.task_file), self.workdir_path)
        self.neural_model = lib.set_neural_model(self.model, reasoning=self.reason)  # type: ignore[attr-defined]

        self.run_id = lib.new_run_id()
        self.paths = get_loop_paths(self.workdir_path)
        self.task_rel = str(self.task_path.relative_to(self.workdir_path))
        self.review_output_rel = str(
            self.paths.review_output_toml.relative_to(self.workdir_path)
        )
        self.system = build_system_prompt(workdir=self.workdir_path)
        self.review_system = build_review_system_prompt()

        self.loop_summaries: List[LoopSummary] = []
        self.review_summaries: List[LoopSummary] = []
        self.items: List[ReviewItem] = []
        self.next_item_id = 1
        self.blocker = ""
        self.author_next_steps: List[str] = []
        self.scores: List[LoopScore] = []
        self.stop_reason = "max_loops"
        self.rules: List[LoopRule] = []
        self.rule_problems: List[str] = []
        self.new_rule_problems: List[str] = []  # from the latest review only
        self.guard = ProtectedFileGuard(self.workdir_path)
        self.loops_completed = 0
        self.task_mtime: float = 0.0
        self.phase_metadata_files: List[Path] = []

        self.load_task_context()
        self.initialize_paths()
        self.write_rules()
        self.initialize_run_record()
        self.initialize_loop_state()

    def load_task_context(self) -> None:
        self.chat_history, meta = lib.load_chat_template(
            self.task_path, return_meta=True
        )
        tools_cfg = meta.get("tools") or {}
        self.bash_allow = set(tools_cfg.get("bash") or [])
        self.memory = tools_cfg.get("memory")
        self.check = meta.get("check")
        self.python_scripts_dir()  # fail fast if memory escapes the workdir
        self.writable = list(tools_cfg.get("writable") or [])
        # The task file is always protected; [tools].protect adds more.
        self.guard.add(self.task_rel)
        for pattern in tools_cfg.get("protect") or []:
            self.guard.add(pattern)
        self.base_protect = list(self.guard.patterns)

    def protected_paths(self) -> set:
        return set(self.guard.resolve()) | {self.task_path.resolve()}

    def make_author_tools(self) -> List[lib.AgentTool]:
        # Rebuilt every phase so reviewer-added protections take effect.
        tools = lib.make_tools(
            self.workdir_path,
            bash_allow=self.bash_allow,
            protected_paths=self.protected_paths(),
            memory=self.memory,
        )
        return [
            GuardedBashTool(t, rules=self.rules, guard=self.guard)
            if t.name == "bash"
            else t
            for t in tools
        ]

    def python_scripts_dir(self) -> str:
        """Workdir-relative dir allowlisted python must run from ("" if no python)."""
        allowed = set(lib.BashTool._DEFAULT_ALLOWED) | self.bash_allow
        path = lib.resolve_script_dir(self.workdir_path, allowed, self.memory)
        return str(path.relative_to(self.workdir_path)) if path else ""

    def write_rules(self) -> None:
        lib.atomic_write_toml(
            self.paths.rules_toml,
            {
                "rule": [r.to_toml() for r in self.rules],
                "protected": [
                    str(p.relative_to(self.workdir_path)) for p in self.guard.resolve()
                ],
                "problems": list(self.rule_problems),
            },
        )

    def initialize_paths(self) -> None:
        os.makedirs(self.paths.run_dir, exist_ok=True)
        lib.ensure_loop_metadata_dir(self.paths.run_dir)

    def initialize_run_record(self) -> None:
        self.run_toml_data = {
            "run_id": self.run_id,
            "created_at": lib.iso_utc_now(),
            "workdir": str(self.workdir_path),
            "task_file": str(self.task_path),
            "model": str(self.model),
            "agent_loops": int(self.agent_loops),
            "agent_iterations": int(self.agent_iterations),
            "reason": bool(self.reason),
            "max_tokens": int(getattr(self.neural_model, "max_tokens", 0)),
            "reasoning_config": _reasoning_config_str(self.neural_model),
            "cumulative_input_tokens": 0,
            "cumulative_output_tokens": 0,
            "cumulative_cache_creation_tokens": 0,
            "cumulative_cache_read_tokens": 0,
            "cumulative_cache_creation_5m_tokens": 0,
            "cumulative_cache_creation_1h_tokens": 0,
            "loops_completed": 0,
        }
        lib.atomic_write_toml(self.paths.run_toml, self.run_toml_data)

    def initialize_loop_state(self) -> None:
        self.state = init_state(
            run_id=self.run_id, workdir=self.workdir_path, task_file=self.task_path
        )
        write_state(self.paths.state_toml, self.state)

    def update_state(self, *, loop_index: int, phase: str) -> None:
        self.state["run_id"] = self.run_id
        self.state["loop_index"] = loop_index
        self.state["phase"] = phase
        write_state(self.paths.state_toml, self.state)

    def reload_task_context_if_needed(self) -> None:
        # Only the task content (chat_history) is refreshed here. The bash
        # allowlist / tool policy is fixed once in load_task_context() at
        # construction time and must never be re-derived mid-run: the task
        # file lives inside the agent's own writable root, so re-reading its
        # [tools].bash on every edit would let the agent grant itself new
        # bash commands by rewriting its own task file.
        try:
            current_mtime = self.task_path.stat().st_mtime
        except OSError:
            current_mtime = 0.0
        if current_mtime != self.task_mtime:
            self.chat_history = lib.load_chat_template(self.task_path)
            self.task_mtime = current_mtime

    def persist_run_progress(self, loop_idx: int) -> None:
        update_cumulative_tokens(
            self.run_toml_data, self.loop_summaries + self.review_summaries
        )
        self.run_toml_data["loops_completed"] = loop_idx
        self.run_toml_data["stop_reason"] = self.stop_reason
        self.run_toml_data["score"] = [s.to_toml() for s in self.scores]
        lib.atomic_write_toml(self.paths.run_toml, self.run_toml_data)
        lib.write_loop_manifest(
            self.paths.metadata_dir,
            run_doc=self.run_toml_data,
            phase_files=self.phase_metadata_files,
        )
        self.loops_completed = loop_idx

    def build_author_logging(self) -> lib.ToolLogSink:
        author_log: lib.ToolLogSink = lib.ToolLogToml(path=str(self.paths.author_toml))
        if self.logging is not None:
            extra_log = lib.ToolLogToml(
                path=str(self.logging) if str(self.logging) else None
            )
            author_log = lib.MultiToolLogSink([author_log, extra_log])
        return author_log

    def build_author_agent(self, logging: lib.ToolLogSink) -> lib.Agent:
        return lib.Agent(
            self.neural_model,
            tools=self.make_author_tools(),
            max_iterations=self.agent_iterations,
            show_diagnostics=self.verbose,
            logging=logging,
        )

    def bash_allowed(self) -> List[str]:
        return sorted(set(lib.BashTool._DEFAULT_ALLOWED) | self.bash_allow)

    def run_check(self) -> Optional[CheckResult]:
        if not self.check:
            return None
        result = run_check(
            self.check["command"], cwd=self.workdir_path, timeout=self.check["timeout"]
        )
        if self.verbose:
            print(f"  check `{result.command}`: {'PASS' if result.passed else 'FAIL'}")
        return result

    def run_author_phase(self, loop_idx: int) -> tuple[LoopSummary, str, Optional[str]]:
        if self.verbose:
            print(f"\n▶  loop {loop_idx} [author]")

        self.update_state(loop_index=loop_idx, phase="author")
        lib.atomic_write_text(self.paths.author_toml, "")

        author_log = self.build_author_logging()
        author_agent = self.build_author_agent(author_log)

        author_task = build_author_task(
            workdir=self.workdir_path,
            task_rel=self.task_rel,
            loop_idx=loop_idx,
            agent_loops=self.agent_loops,
            loop_summaries=self.loop_summaries,
            items=self.items,
            blocker=self.blocker,
            rules=self.rules,
            protected=[
                str(p.relative_to(self.workdir_path)) for p in self.guard.resolve()
            ],
            python_scripts_dir=self.python_scripts_dir(),
            bash_allowed=self.bash_allowed(),
        )

        self.guard.snapshot()
        tree_before = workspace_tree(self.workdir_path)
        phase_timer = lib.Timer()
        author_result = author_agent.run(
            author_task, system=self.system, chat_history=self.chat_history
        )
        phase_duration_s = phase_timer.ms / 1000.0
        # Catch-all for changes not made through bash (e.g. background processes).
        self.guard.restore_changes()
        tree_after = workspace_tree(self.workdir_path)
        author_answer = author_result.final_text or ""
        loop_summary = loop_summary_from_result(loop_idx, author_result)
        loop_summary.violations = list(self.guard.violations)
        loop_summary.diff = workspace_diff(self.workdir_path, tree_before, tree_after)
        loop_summary.changed = (
            tree_before != tree_after
            if tree_before
            else bool(loop_summary.files_written or loop_summary.files_edited)
        )
        loop_summary.check = self.run_check()
        if self.verbose and loop_summary.violations:
            print("  ⚠ protected files reverted: " + "; ".join(loop_summary.violations))

        self.loop_summaries.append(loop_summary)
        self.author_next_steps = extract_pending_items(author_answer)
        self.phase_metadata_files.append(
            lib.write_loop_phase_metadata(
                self.paths.metadata_dir,
                run_id=self.run_id,
                loop_index=loop_idx,
                phase="author",
                model=str(self.model),
                task_file=str(self.task_path),
                workdir=str(self.workdir_path),
                stop_reason=author_result.stop_reason,
                final_text_present=bool(author_result.final_text),
                usage=author_result.usage,
                iterations=author_result.iterations,
                tool_results=author_result.tool_results,
                rejected_calls=author_result.rejected_calls,
                duration_s=phase_duration_s,
            )
        )
        self.persist_run_progress(loop_idx)
        return loop_summary, author_answer, extract_status(author_answer)

    def build_review_tools(self) -> List[lib.AgentTool]:
        review_bash_allow = {
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
        protected = self.protected_paths()
        review_tools = [
            t
            for t in lib.make_tools(self.workdir_path, protected_paths=protected)
            if t.name in ("read", "glob", "write")
        ]
        review_tools.append(
            lib.BashTool(
                cwd=self.workdir_path,
                bounded=True,
                allowed_commands=review_bash_allow,
                protected_paths=protected,
            )
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

    def apply_review_data(self, review_data: Dict[str, Any], loop_idx: int) -> int:
        """Apply the reviewer's output; return the number of resolved items."""
        self.blocker = str(review_data.get("blocker") or "").strip()
        self.apply_review_rules(review_data)
        return self.update_items(review_data.get("pending"), loop_idx)

    def apply_review_rules(self, review_data: Dict[str, Any]) -> None:
        """Retire and add reviewer rules."""
        retire = review_data.get("retire") or []
        if isinstance(retire, list):
            retire_ids = {str(x).strip() for x in retire}
            self.rules[:] = [r for r in self.rules if r.id not in retire_ids]

        loop_no = review_data.get("loop", "?")
        next_id = (
            max((int(r.id[1:]) for r in self.rules if r.id[1:].isdigit()), default=0)
            + 1
        )
        new_rules, problems = parse_rules(
            review_data.get("rule"), source=f"review:{loop_no}", start_id=next_id
        )
        # A rule may not lock a file the author must change (e.g. its work log):
        # drop the protect, keep the text.
        for rule in new_rules:
            if not rule.protect:
                continue
            hit = protect_conflict(rule.protect, self.writable, self.workdir_path)
            if hit:
                problems.append(
                    f"rule protect {rule.protect!r} targets author-writable {hit!r}; ignored"
                )
                rule.protect = ""
        known = {r.text for r in self.rules}
        self.rules.extend(r for r in new_rules if r.text not in known)
        self.rule_problems.extend(problems)
        self.new_rule_problems = problems

        del self.rules[: max(0, len(self.rules) - MAX_REVIEW_RULES)]

        # [tools].protect and the task file stay protected for the whole run, so a
        # confused reviewer cannot reopen a human-owned path. A rule's protect
        # lasts only while the rule is active, so retiring a rule undoes a
        # protection that turned out to block required work.
        self.guard.patterns = list(self.base_protect)
        for rule in self.rules:
            self.guard.add(rule.protect)
        self.write_rules()

    def run_review_phase(
        self, loop_idx: int, loop_summary: LoopSummary, exec_answer: str
    ) -> Optional[int]:
        """Run the reviewer; return resolved-item count, or None if it wrote no output."""
        if self.verbose:
            print(f"\n▶  loop {loop_idx} [review]")

        self.update_state(loop_index=loop_idx, phase="review")
        self.paths.review_output_toml.unlink(missing_ok=True)
        review_agent = self.build_review_agent()

        review_task = build_review_task(
            workdir=self.workdir_path,
            task_rel=self.task_rel,
            review_output_rel=self.review_output_rel,
            loop_index=loop_idx,
            loop_summary=loop_summary,
            exec_answer=exec_answer or "",
            items=self.items,
            rules=self.rules,
            protected=[
                str(p.relative_to(self.workdir_path)) for p in self.guard.resolve()
            ],
            rule_problems=self.new_rule_problems,
        )

        phase_timer = lib.Timer()
        review_result = review_agent.run(
            review_task, system=self.review_system, chat_history=self.chat_history
        )
        phase_duration_s = phase_timer.ms / 1000.0
        review_loop_summary = loop_summary_from_result(loop_idx, review_result)
        self.review_summaries.append(review_loop_summary)
        self.phase_metadata_files.append(
            lib.write_loop_phase_metadata(
                self.paths.metadata_dir,
                run_id=self.run_id,
                loop_index=loop_idx,
                phase="review",
                model=str(self.model),
                task_file=str(self.task_path),
                workdir=str(self.workdir_path),
                stop_reason=review_result.stop_reason,
                final_text_present=bool(review_result.final_text),
                usage=review_result.usage,
                iterations=review_result.iterations,
                tool_results=review_result.tool_results,
                rejected_calls=review_result.rejected_calls,
                duration_s=phase_duration_s,
            )
        )
        self.persist_run_progress(loop_idx)

        review_data = read_review_output(self.paths.review_output_toml)
        if review_data:
            return self.apply_review_data(review_data, loop_idx)
        # No review: fall back to the author's own next steps.
        self.blocker = ""
        self.update_items(self.author_next_steps, loop_idx)
        return None

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
            self.reload_task_context_if_needed()
            summary, author_answer, status = self.run_author_phase(loop_idx)
            check_ok = summary.check is None or summary.check.passed

            # Self-reported completion only counts when a check confirms it.
            if status == "COMPLETE" and summary.check and summary.check.passed:
                self.items, self.blocker = [], ""
                self.stop("complete", f"author COMPLETE and check passed (loop {loop_idx})")
                break

            resolved = self.run_review_phase(loop_idx, summary, author_answer)
            self.record_score(summary, resolved or 0)
            self.persist_run_progress(loop_idx)

            if resolved is not None and check_ok and not self.items and not self.blocker:
                self.stop("complete", f"review clean and check passed (loop {loop_idx})")
                break
            if self.stalled():
                self.stop("stalled", f"no progress for {self.stall_loops} loops (loop {loop_idx})")
                break

        self.persist_run_progress(self.loops_completed)
        return (
            f"{self.stop_reason}: {self.loops_completed}/{self.agent_loops} loop(s) "
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
    summary.total_cache_creation_5m_tokens = u.cache_write_5m
    summary.total_cache_creation_1h_tokens = u.cache_write_1h

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


def build_review_system_prompt() -> str:
    return (
        "You are a code reviewer verifying another agent's work against a task specification.\n\n"
        "Core rules:\n"
        "- Judge from the harness evidence (diff, check output) and the files — never from the\n"
        "  author's claims alone.\n"
        "- NEVER fabricate file contents or results — read the file when the diff is not enough.\n"
        "- Report only concrete defects or missing work that stand between the code and the\n"
        "  specification. No style nits, no speculative improvements.\n"
        "- Be decisive: when the specification is met, say so and raise nothing.\n"
        "- Only access files within the working directory.\n"
    )


def format_check(check: Optional[CheckResult], *, tail_on_pass: bool = False) -> str:
    """Render a check result; the output tail is shown on failure (or always, if asked)."""
    if check is None:
        return ""
    verdict = "PASSED" if check.passed else f"FAILED (exit {check.exit_code})"
    line = f"Harness check `{check.command}`: {verdict}"
    if check.tail and (tail_on_pass or not check.passed):
        line += f"\n```text\n{check.tail}\n```"
    return line


def format_items(items: List[ReviewItem]) -> List[str]:
    return [f"  - [{i.id}, open since loop {i.opened}] {i.text}" for i in items]


def format_loop_context(
    *,
    loop_idx: int,
    agent_loops: int,
    loop_summaries: List[LoopSummary],
    items: List[ReviewItem],
    blocker: str = "",
) -> str:
    """Build the injected context block that orients the author agent."""
    lines = [f"Loop {loop_idx} of {agent_loops}."]

    written = list(dict.fromkeys(f for s in loop_summaries for f in s.files_written))
    edited = list(dict.fromkeys(f for s in loop_summaries for f in s.files_edited))
    if written:
        lines.append("Files created across all prior loops: " + ", ".join(written))
    if edited:
        lines.append("Files edited across all prior loops: " + ", ".join(edited))

    if loop_summaries:
        last = loop_summaries[-1]
        if last.check:
            lines.append(format_check(last.check))
        if last.errors:
            lines.append("Last loop errors: " + "; ".join(last.errors[:5]))
        if last.violations:
            lines.append(
                "VIOLATION last loop — you changed protected files and the harness "
                "reverted them: " + "; ".join(last.violations) + ". Do not repeat this."
            )

    if blocker:
        lines.append(f"Reviewer blocker: {blocker}")
    if items:
        lines.append("Open review items (resolve every one):")
        lines.extend(format_items(items))
    elif loop_idx > 1:
        lines.append("No open review items — determine the next action from the task.")

    return "\n".join(lines)


def format_policy_block(rules: List[LoopRule], protected: List[str]) -> str:
    """Render protected files and binding rules for either phase's task."""
    if not rules and not protected:
        return ""
    lines = ["HARNESS-ENFORCED POLICY"]
    if protected:
        lines.append(
            "Protected files (read-only; any change is reverted and recorded as a "
            "violation): " + ", ".join(protected)
        )
    if rules:
        lines.append(
            "Binding rules (they override pending steps and any instruction to keep "
            "going):"
        )
        for r in rules:
            extra = f" [bash matching /{r.deny_bash}/ is refused]" if r.deny_bash else ""
            lines.append(f"- {r.id} ({r.source}): {r.text}{extra}")
    return "\n".join(lines)


def build_author_task(
    *,
    workdir: Path,
    task_rel: str,
    loop_idx: int,
    agent_loops: int,
    loop_summaries: List[LoopSummary],
    items: List[ReviewItem],
    blocker: str = "",
    rules: Optional[List[LoopRule]] = None,
    protected: Optional[List[str]] = None,
    python_scripts_dir: str = "",
    bash_allowed: Optional[List[str]] = None,
) -> str:
    context = format_loop_context(
        loop_idx=loop_idx,
        agent_loops=agent_loops,
        loop_summaries=loop_summaries,
        items=items,
        blocker=blocker,
    )
    policy = format_policy_block(rules or [], protected or [])
    if policy:
        context = f"{context}\n\n{policy}"
    if loop_idx == 1:
        orient_step = (
            "1. The task specification is the conversation above — do NOT re-read the task file.\n"
        )
    else:
        orient_step = (
            "1. The specification is the conversation above and prior work is summarised —\n"
            "   do NOT re-read the task file or re-glob the workspace to orient yourself.\n"
        )
    if python_scripts_dir:
        fallback_check = (
            f"   If none exists, write a check script under {python_scripts_dir}/ (e.g. one\n"
            "   calling compileall) and run it.\n"
            f"   Python only runs script files under {python_scripts_dir}/ (no -c/-m/stdin):\n"
            "   write every python snippet there first so it is kept for provenance.\n"
        )
    else:
        fallback_check = "   If none exists, run `python -m compileall .`.\n"
    bash_line = (
        "Allowed bash commands (no pipes, redirects or shell metacharacters): "
        + ", ".join(bash_allowed)
        + "\n"
        if bash_allowed
        else ""
    )
    return (
        f"{context}\n\n"
        f"Working directory: {workdir}\n"
        f"Task file: {task_rel} (read-only)\n"
        f"{bash_line}\n"
        "PHASE: AUTHOR\n\n"
        "Goal: COMPLETE the task in this single session if at all possible. Implement every\n"
        "remaining item you can, verify it, and only stop when the task is fully done or you\n"
        "are genuinely blocked. Do NOT implement just one item and defer the rest to a later\n"
        "loop — keep working until everything that can be done this session is done.\n"
        "The harness runs its own check after you finish; your report cannot override it.\n\n"
        "Protocol:\n"
        + orient_step
        + "2. Write a short PLAN (3–7 bullets) covering everything you intend to complete now.\n"
        "3. Execute the plan autonomously and to completion — do NOT ask for confirmation, and\n"
        "   do NOT stop after a single change while more work remains.\n"
        "4. Before each set of tool calls, write one or two sentences stating what you are about to do and why.\n"
        "5. Inspect the current state of relevant files before editing them.\n"
        "6. After each meaningful change, run the closest available check (tests, lint, typecheck).\n"
        + fallback_check
        + "   Do not defer all validation to the end.\n"
        "7. Continue until every task is implemented AND the checks pass, or you hit a hard blocker.\n"
        "   A stop condition in the task file or a binding rule (e.g. waiting for human\n"
        "   approval) IS a hard blocker: stop, report STATUS: INCOMPLETE, and name it in\n"
        "   NEXT STEPS. Never work around it.\n\n"
        "Finish with <final_answer> that includes, in this order:\n"
        "- A status line of the EXACT form `STATUS: COMPLETE` if every task is implemented and all\n"
        "  checks pass, otherwise `STATUS: INCOMPLETE`.\n"
        "- The PLAN you followed (final form), naming the review item ids you resolved.\n"
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
    items: Optional[List[ReviewItem]] = None,
    rules: Optional[List[LoopRule]] = None,
    protected: Optional[List[str]] = None,
    rule_problems: Optional[List[str]] = None,
) -> str:
    # Harness-verified facts, so the reviewer never has to trust the transcript.
    summary_lines = [f"## Verified actions from loop {loop_index} (harness-computed)"]
    if loop_summary.files_read:
        summary_lines.append("Files read: " + ", ".join(loop_summary.files_read))
    if loop_summary.files_written:
        summary_lines.append("Files written: " + ", ".join(loop_summary.files_written))
    if loop_summary.files_edited:
        summary_lines.append("Files edited: " + ", ".join(loop_summary.files_edited))
    if loop_summary.commands_run:
        summary_lines.append("Commands run:")
        summary_lines.extend(f"  {c}" for c in loop_summary.commands_run)
    if loop_summary.errors:
        summary_lines.append("Errors:")
        summary_lines.extend(f"  {e}" for e in loop_summary.errors)
    if len(summary_lines) == 1:
        summary_lines.append("  (no verified actions)")
    blocks = ["\n".join(summary_lines)]

    if loop_summary.check:
        blocks.append("## " + format_check(loop_summary.check, tail_on_pass=True))
    if loop_summary.diff:
        blocks.append(f"## Workspace diff from loop {loop_index}\n```diff\n{loop_summary.diff}```")
    elif not loop_summary.changed:
        blocks.append("## Workspace diff: the author changed nothing.")

    # Rejected calls never touched the workspace; claims built on them are unverified.
    if loop_summary.rejected:
        blocks.append(
            "\n".join(
                ["## Attempted but NOT executed (harness-rejected — no workspace effect)"]
                + [f"  {r}" for r in loop_summary.rejected]
            )
        )
    if loop_summary.violations:
        blocks.append(
            "\n".join(
                ["## Protected-file violations (harness reverted these changes)"]
                + [f"  {v}" for v in loop_summary.violations]
            )
        )
    if items:
        blocks.append("\n".join(["## Open review items from earlier loops"] + format_items(items)))
    policy_block = format_policy_block(rules or [], protected or [])
    if policy_block:
        blocks.append(policy_block)
    if rule_problems:
        blocks.append(
            "\n".join(
                ["## Your previous rules, partly rejected by the harness"]
                + [f"  {x}" for x in rule_problems]
            )
        )
    if exec_answer:
        blocks.append(f"[Author agent report — loop {loop_index}]\n\n{exec_answer}")

    evidence = "\n\n".join(blocks)
    return (
        f"Working directory: {workdir}\n"
        f"Task file: {task_rel} (read-only; its specification is the conversation above)\n"
        f"Review output: {review_output_rel} (write here)\n\n"
        "PHASE: REVIEW\n\n"
        f"{evidence}\n\n"
        "Your job:\n"
        "1. Judge the work against the specification using the diff, the harness check and the\n"
        "   files. Claims relying on an 'Attempted but NOT executed' call did not happen.\n"
        "2. Write your assessment to the review output file in TOML format:\n\n"
        "   ```toml\n"
        f"   loop = {loop_index}\n"
        '   summary = "One paragraph describing what actually happened."\n'
        '   blocker = "Main current blocker, or empty string if none."\n'
        '   retire = []  # optional ids of earlier review rules that no longer apply\n\n'
        "   [[pending]]\n"
        '   id = "P1"    # reuse an open item\'s id to keep it open; omit for a new item\n'
        '   item = "Concrete defect or missing work"\n\n'
        "   # Optional: binding rules for the author, enforced from the next loop on.\n"
        "   [[rule]]\n"
        '   text = "What the author must or must not do, and why"\n'
        '   deny_bash = ""   # optional regex; matching author bash commands are refused\n'
        '   protect = ""     # optional path the author may NOT change at all (read-only)\n'
        "   ```\n\n"
        "   Open items you do not restate are treated as resolved — restate only those the\n"
        "   evidence shows are still unresolved.\n"
        "3. Add a [[rule]] only for a harmful or repeated deviation from the task file or the\n"
        "   binding rules. Prefer deny_bash/protect: those are enforced by the harness, plain\n"
        "   text is not. Do not re-add a rule that is already listed.\n"
        "   `protect` makes a file read-only for the author: never set it on a file the task\n"
        "   requires the author to create or change (e.g. a log it must append to), even if\n"
        "   the rule is about that file. If the author was blocked from required work by a\n"
        "   rule (a protected path or refused command), retire that rule.\n\n"
        "Rules:\n"
        "- A failing harness check, a protected-file violation or a binding-rule breach is a blocker.\n"
        "- If the check passes (or none is configured) and the specification is met, set\n"
        '  blocker = "" and list no pending items.\n'
        "- Pending items must be concrete defects or missing work, not 'continue working'.\n"
        "- Never put an action into pending that a task-file stop condition or a binding rule forbids.\n"
        f"- At most {MAX_ITEMS} pending items.\n\n"
        "After writing the review output file, reply with <final_answer> as plain text.\n"
        "Do not call any more tools to confirm the write.\n"
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
    """Parse NEXT STEPS: section from an author agent's final_answer."""
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
            else:
                digit_count = 0
                while digit_count < len(stripped) and stripped[digit_count].isdigit():
                    digit_count += 1
                if digit_count > 0:
                    rest = stripped[digit_count:]
                    if rest.startswith(". ") or rest.startswith(") "):
                        items.append(rest[2:].strip())
    return items[:5]


def extract_status(final_text: str) -> Optional[str]:
    """Parse a `STATUS: COMPLETE` / `STATUS: INCOMPLETE` line from a final answer.

    Returns "COMPLETE", "INCOMPLETE", or None if no status line was emitted.
    """
    for line in (final_text or "").splitlines():
        s = line.strip().upper()
        if s.startswith("STATUS:"):
            val = s[len("STATUS:") :].strip()
            if val.startswith("COMPLETE"):
                return "COMPLETE"
            if val.startswith("INCOMPLETE"):
                return "INCOMPLETE"
    return None


def _reasoning_config_str(neural_model: Any) -> str:
    thinking = getattr(neural_model, "thinking", None)
    if thinking:
        return "thinking=" + ",".join(f"{k}:{v}" for k, v in sorted(thinking.items()))
    effort = getattr(neural_model, "reasoning_effort", None)
    if effort:
        return f"reasoning_effort={effort}"
    return "provider-default"


def update_cumulative_tokens(
    run_toml_data: Dict[str, Any], summaries: List[LoopSummary]
) -> None:
    """Roll up per-loop token totals into the run.toml record (in place)."""
    run_toml_data["cumulative_input_tokens"] = sum(
        s.total_input_tokens for s in summaries
    )
    run_toml_data["cumulative_output_tokens"] = sum(
        s.total_output_tokens for s in summaries
    )
    run_toml_data["cumulative_cache_creation_tokens"] = sum(
        s.total_cache_creation_tokens for s in summaries
    )
    run_toml_data["cumulative_cache_read_tokens"] = sum(
        s.total_cache_read_tokens for s in summaries
    )
    run_toml_data["cumulative_cache_creation_5m_tokens"] = sum(
        s.total_cache_creation_5m_tokens for s in summaries
    )
    run_toml_data["cumulative_cache_creation_1h_tokens"] = sum(
        s.total_cache_creation_1h_tokens for s in summaries
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
    """Run a bounded author → review loop.

    Cross-loop state is carried entirely in-memory (loop_summaries, items).
    Agents receive context injected into their task strings — they do not read
    state or history files to orient themselves.

    Per loop:
      1) AUTHOR agent: does work + reports STATUS + NEXT STEPS; the harness then
         records the workspace diff and runs [check]. STATUS: COMPLETE with a
         passing check ends the run (review skipped).
      2) REVIEW agent: receives the diff, check output and action summary +
         writes review_output.toml. A clean review with a passing check ends the
         run; `stall_loops` loops without progress also end it.

    Persistent files under `.codescribe/loop/` (for inspection / crash-resume):
      run.toml             — run configuration
      state.toml           — current loop index + phase
      author.toml          — raw TOML event log for the most recent author phase
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
