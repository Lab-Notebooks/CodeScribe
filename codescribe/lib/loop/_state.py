# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Loop state: on-disk paths, rules, review items, scores and phase summaries."""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from codescribe.lib.agent import RunResult
from codescribe.lib.util import atomic_write_toml, iso_utc_now
from codescribe.lib.loop._check import CheckResult

MAX_ITEMS = 8  # open review items carried forward
MAX_REVIEW_RULES = 12  # oldest rules drop first, bounding the author prompt
_PHASES = frozenset({"idle", "author", "review"})


@dataclass
class LoopPaths:
    """Files under `.codescribe/loop/`."""

    run_dir: Path
    metadata_dir: Path
    run_toml: Path
    state_toml: Path
    author_toml: Path  # author event log, rewritten each author phase
    review_output_toml: Path  # reviewer's structured output, rewritten each review
    rules_toml: Path  # active rules, written for inspection only


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


def init_state(*, run_id: str, workdir: Path, task_file: Path) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "workdir": str(workdir),
        "task_file": str(task_file),
        "loop_index": 0,
        "phase": "idle",
        "updated_at": iso_utc_now(),
    }


def write_state(path: Path, state: Dict[str, Any]) -> None:
    run_id, loop_index, phase = (
        state.get("run_id"),
        state.get("loop_index"),
        state.get("phase"),
    )
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"state['run_id'] must be a non-empty str, got {run_id!r}")
    if not isinstance(loop_index, int) or isinstance(loop_index, bool):
        raise ValueError(f"state['loop_index'] must be an int, got {loop_index!r}")
    if phase not in _PHASES:
        raise ValueError(
            f"state['phase'] must be one of {sorted(_PHASES)}, got {phase!r}"
        )
    atomic_write_toml(path, {**state, "updated_at": iso_utc_now()})


@dataclass
class LoopRule:
    """A reviewer-added constraint on the author.

    `text` is always shown to the author; `deny_bash` (regex) refuses matching
    bash commands; `protect` (path or glob) makes files read-only.
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


def parse_rules(
    raw: Any, *, source: str, start_id: int
) -> Tuple[List[LoopRule], List[str]]:
    """Rules from reviewer entries (strings or tables), plus problems found.

    Invalid or match-everything `deny_bash` patterns are dropped (the text is
    kept) so a bad regex cannot brick the author.
    """
    rules: List[LoopRule] = []
    problems: List[str] = []
    for entry in raw if isinstance(raw, list) else []:
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
                    problems.append(
                        f"deny_bash {deny!r} matches every command; ignored"
                    )
                    deny = ""
            except re.error as exc:
                problems.append(
                    f"deny_bash {deny!r} is not a valid regex ({exc}); ignored"
                )
                deny = ""
        rules.append(
            LoopRule(
                id=f"R{start_id + len(rules)}",
                text=text,
                deny_bash=deny,
                protect=str(entry.get("protect") or "").strip(),
                source=source,
            )
        )
    return rules, problems


def protect_conflict(pattern: str, writable: List[str], workdir: Path) -> str:
    """The author-writable entry a `protect` pattern would lock, or ""."""
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
    for w in map(os.path.normpath, writable):
        if fnmatch.fnmatch(w, pattern) or any(fnmatch.fnmatch(h, w) for h in hits):
            return w
    return ""


@dataclass
class ReviewItem:
    """An open review finding; its id is stable across loops."""

    id: str
    text: str
    opened: int  # loop that first raised it


@dataclass
class LoopScore:
    loop_index: int
    changed: bool
    check_passed: Optional[bool]
    resolved: int
    open_items: int
    progress: bool

    def to_toml(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _call_label(name: str, args: Any) -> str:
    args = args if isinstance(args, dict) else {}
    if name == "bash":
        return (args.get("command") or "").replace("\n", " ").strip()[:80]
    if name in ("read", "write", "edit"):
        return str(args.get("path", ""))
    return ""


@dataclass
class LoopSummary:
    """What one phase did, computed by the harness from the agent's RunResult."""

    loop_index: int
    files_written: List[str] = field(default_factory=list)
    files_edited: List[str] = field(default_factory=list)
    files_read: List[str] = field(default_factory=list)
    commands_run: List[str] = field(default_factory=list)  # "cmd  →  first output line"
    errors: List[str] = field(default_factory=list)
    rejected: List[str] = field(
        default_factory=list
    )  # refused by the harness, never executed
    violations: List[str] = field(
        default_factory=list
    )  # protected-file changes reverted
    changed: bool = False
    diff: str = ""
    check: Optional[CheckResult] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_5m_tokens: int = 0
    total_cache_creation_1h_tokens: int = 0

    @classmethod
    def from_result(cls, loop_index: int, result: RunResult) -> "LoopSummary":
        u = result.usage
        summary = cls(
            loop_index=loop_index,
            total_input_tokens=u.input,
            total_output_tokens=u.output,
            total_cache_creation_tokens=u.cache_write,
            total_cache_read_tokens=u.cache_read,
            total_cache_creation_5m_tokens=u.cache_write_5m,
            total_cache_creation_1h_tokens=u.cache_write_1h,
        )
        for tr in result.tool_results:
            tool = tr.name or "?"
            label = _call_label(tool, tr.args)
            preview = (tr.output_preview or "").strip()
            if not tr.ok:
                summary.errors.append(
                    f"{tool}({label}): {preview[:80] or 'unknown error'}"
                )
            elif tool == "write":
                summary.files_written.append(label)
            elif tool == "edit":
                summary.files_edited.append(label)
            elif tool == "read":
                summary.files_read.append(label)
            elif tool == "bash":
                summary.commands_run.append(
                    f"{label}  →  {(preview.splitlines() or [''])[0][:60]}"
                )
        for rc in result.rejected_calls:
            tool = rc.name or "?"
            summary.rejected.append(
                f"{tool}({_call_label(tool, rc.args)}): {rc.reason}"
            )
        return summary


_TOKEN_TOTALS = {
    "cumulative_input_tokens": "total_input_tokens",
    "cumulative_output_tokens": "total_output_tokens",
    "cumulative_cache_creation_tokens": "total_cache_creation_tokens",
    "cumulative_cache_read_tokens": "total_cache_read_tokens",
    "cumulative_cache_creation_5m_tokens": "total_cache_creation_5m_tokens",
    "cumulative_cache_creation_1h_tokens": "total_cache_creation_1h_tokens",
}


def update_cumulative_tokens(
    run_doc: Dict[str, Any], summaries: List[LoopSummary]
) -> None:
    """Roll phase token totals up into the run.toml record (in place)."""
    for key, attr in _TOKEN_TOTALS.items():
        run_doc[key] = sum(getattr(s, attr) for s in summaries)
