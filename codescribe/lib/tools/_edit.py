# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from codescribe.lib.tools._base import PathTool, schema


def snippet(text: str, start: int, end: int, context: int = 80) -> str:
    """`text[start:end]` with `context` chars either side."""
    left, right = max(0, start - context), min(len(text), end + context)
    return ("…" if left else "") + text[left:right] + ("…" if right < len(text) else "")


class EditTool(PathTool):
    def __init__(
        self, root: Optional[Path] = None, protected_paths: Optional[set] = None
    ) -> None:
        super().__init__(
            "edit",
            "Edit a file using exact text replacements (all oldText must match exactly once).",
            schema(
                ["path", "edits"],
                path={"type": "string", "description": "Path to the file to edit."},
                edits={
                    "type": "array",
                    "description": "List of replacements.",
                    "items": schema(
                        ["oldText", "newText"],
                        oldText={"type": "string"},
                        newText={"type": "string"},
                    ),
                    "minItems": 1,
                },
            ),
            root=root,
            protected_paths=protected_paths,
        )

    def run(self, args: Dict[str, Any]) -> str:
        path = args.get("path")
        edits = args.get("edits")
        if not path:
            return "Error: missing required argument 'path'"
        if not edits:
            return "Error: missing required argument 'edits'"
        try:
            path = self.resolve(path)
        except Exception as exc:
            return f"Error: {exc}"
        err = self.write_error(path, "editing")
        if err:
            return err
        if not os.path.exists(path):
            return f"Error: file not found: {path}"
        if not os.path.isfile(path):
            return f"Error: not a file: {path}"
        try:
            with open(path, "r", errors="replace") as fh:
                before = fh.read()
        except Exception as exc:
            return f"Error: {exc}"

        # Every oldText must be unique in the original content.
        for e in edits:
            old, new = e.get("oldText"), e.get("newText")
            if old is None or new is None:
                return "Error: each edit must include oldText and newText"
            count = before.count(old)
            if count != 1:
                return f"Error: oldText must match exactly once; found {count} matches for: {old!r}"

        after = before
        for e in edits:
            after = after.replace(e["oldText"], e["newText"])
        try:
            with open(path, "w") as fh:
                fh.write(after)
        except Exception as exc:
            return f"Error: {exc}"

        replacements = []
        for idx, e in enumerate(edits):
            old, new = e["oldText"], e["newText"]
            start = before.find(old)
            pos = after.find(new)
            replacements.append(
                {
                    "index": idx,
                    "match_count": 1,
                    "old_len": len(old),
                    "new_len": len(new),
                    "before_snippet": snippet(before, start, start + len(old)),
                    "after_snippet": (
                        snippet(after, pos, pos + len(new))
                        if pos >= 0
                        else "(newText not found after write)"
                    ),
                }
            )
        report = {
            "ok": True,
            "path": path,
            "applied": len(edits),
            "replacements": replacements,
        }
        return json.dumps(report, ensure_ascii=False, indent=2)
