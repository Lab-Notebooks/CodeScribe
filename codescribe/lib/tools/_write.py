# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from codescribe.lib.tools._base import PathTool, schema


class WriteTool(PathTool):
    def __init__(
        self, root: Optional[Path] = None, protected_paths: Optional[set] = None
    ) -> None:
        super().__init__(
            "write",
            "Write a file (create or overwrite).",
            schema(
                ["path", "content"],
                path={"type": "string", "description": "Path to write."},
                content={"type": "string", "description": "Full content to write."},
            ),
            root=root,
            protected_paths=protected_paths,
        )

    def run(self, args: Dict[str, Any]) -> str:
        path = args.get("path")
        content = args.get("content")
        if not path:
            return "Error: missing required argument 'path'"
        if content is None:
            return "Error: missing required argument 'content'"
        try:
            path = self.resolve(path)
        except Exception as exc:
            return f"Error: {exc}"
        err = self.write_error(path, "writing")
        if err:
            return err

        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w") as fh:
                fh.write(content)
            return f"Wrote {path} ({len(content)} bytes)"
        except Exception as exc:
            return f"Error: {exc}"
