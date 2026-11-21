# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Convergence signals: the task's [check] command and the workspace diff."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

CHECK_TAIL_CHARS = 4000
MAX_DIFF_CHARS = 20000


@dataclass
class CheckResult:
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
    return subprocess.run(
        ["git", *args],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    ).stdout


def workspace_tree(workdir: Path) -> str:
    """Git tree hash of the workdir (untracked files included), or "" outside git.

    Uses a throwaway index so the user's staging area is untouched.
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
        diff = (
            diff[:MAX_DIFF_CHARS] + "\n[diff truncated; read the files for the rest]\n"
        )
    return diff
