# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any, Dict, Optional

from codescribe.lib.tools._base import PathTool, schema


class GlobTool(PathTool):
    def __init__(self, root: Optional[Path] = None) -> None:
        super().__init__(
            "glob",
            "List files matching a glob pattern (e.g. '**/*.py'). "
            "Returns newline-separated paths (relative to root when bounded).",
            schema(
                ["pattern"],
                pattern={
                    "type": "string",
                    "description": "Glob pattern (supports **).",
                },
                root={
                    "type": "string",
                    "description": "Optional root directory for the search (default: current/bounded root).",
                },
                include_dirs={
                    "type": "integer",
                    "description": "Set to 1 to include directories; 0/omitted to include only files.",
                },
                limit={
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "minimum": 1,
                },
            ),
            root=root,
        )

    def run(self, args: Dict[str, Any]) -> str:
        pattern = args.get("pattern")
        root_arg = args.get("root")
        include_dirs = bool(int(args.get("include_dirs") or 0))
        limit = int(args.get("limit") or 2000)
        if not pattern:
            return "Error: missing required argument 'pattern'"

        if self.root is None:
            base = Path(root_arg).resolve() if root_arg else Path.cwd().resolve()
        else:
            try:
                base = Path(self.resolve(root_arg)) if root_arg else self.root
            except Exception as exc:
                return f"Error: {exc}"

        try:
            matches = glob(str(base / pattern), recursive=True)
        except Exception as exc:
            return f"Error: {exc}"

        out = set()
        for m in matches:
            try:
                p = Path(self.resolve(m)).resolve()
            except Exception:
                continue  # escaped the bounded root
            if not include_dirs and p.is_dir():
                continue
            try:
                out.add(str(p.relative_to(base)))
            except ValueError:
                out.add(str(p))
        return "\n".join(sorted(out)[:limit])
