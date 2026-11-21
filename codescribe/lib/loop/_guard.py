# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Protected-file enforcement for the author agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from codescribe.lib.tools import AgentTool, ensure_within_workdir
from codescribe.lib.loop._state import LoopRule


class ProtectedFileGuard:
    """Snapshot protected files and revert any change to them.

    Path checks cannot see inside an allowlisted interpreter (`python3 x.py`),
    so contents are compared after every author bash call and at phase end.
    Tamper repair, not an OS sandbox.
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

    def relative(self) -> List[str]:
        return [str(p.relative_to(self.workdir)) for p in self.resolve()]

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


class GuardedBashTool(AgentTool):
    """Author bash: refuses rule-denied commands and reverts protected-file changes."""

    def __init__(
        self, inner: AgentTool, *, rules: List[LoopRule], guard: ProtectedFileGuard
    ) -> None:
        super().__init__(inner.name, inner.description, inner.parameters, inner.enabled)
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
                f"them and recorded a violation: {'; '.join(reverted)}\n\n{output}"
            )
        return output
