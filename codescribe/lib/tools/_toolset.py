# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Bounded toolsets rooted at a working directory."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from codescribe.lib.tools._base import AgentTool
from codescribe.lib.tools._bash import BashTool
from codescribe.lib.tools._edit import EditTool
from codescribe.lib.tools._glob import GlobTool
from codescribe.lib.tools._read import ReadTool
from codescribe.lib.tools._write import WriteTool

# Memory layer: workdir-relative home for durable agent artifacts ([tools].memory).
DEFAULT_MEMORY_DIR = ".codescribe/mem"
# Allowlisted python may only create scripts under <mem>/scripts.
SCRIPTS_SUBDIR = "scripts"


def resolve_script_dir(
    root: Path, allowed: set, memory: Optional[str]
) -> Optional[Path]:
    """`<memory>/scripts` if python is allowlisted, else None; raises if it escapes `root`."""
    if not any(BashTool.is_python(cmd) for cmd in allowed):
        return None
    mem = memory or DEFAULT_MEMORY_DIR
    return AgentTool.resolve_within_root(root, str(Path(mem) / SCRIPTS_SUBDIR))


def make_tools(
    root: Path,
    bash_allow: Optional[set] = None,
    protected_paths: Optional[set] = None,
    memory: Optional[str] = None,
) -> List[AgentTool]:
    """read/glob/bash/edit/write bounded to `root`.

    `bash_allow` extends the bash allowlist; `protected_paths` may never be
    written by any tool.
    """
    allowed = BashTool.DEFAULT_ALLOWED | set(bash_allow or ())
    return [
        ReadTool(root=root),
        GlobTool(root=root),
        BashTool(
            cwd=root,
            bounded=True,
            allowed_commands=allowed,
            protected_paths=protected_paths,
            script_dir=resolve_script_dir(root, allowed, memory),
        ),
        EditTool(root=root, protected_paths=protected_paths),
        WriteTool(root=root, protected_paths=protected_paths),
    ]


def make_readonly_tools(
    root: Path,
    bash_allow: Optional[set] = None,
    protected_paths: Optional[set] = None,
    memory: Optional[str] = None,
) -> List[AgentTool]:
    """read/glob/bash bounded to `root`."""
    tools = make_tools(root, bash_allow, protected_paths, memory)
    return [t for t in tools if t.name in ("read", "glob", "bash")]
