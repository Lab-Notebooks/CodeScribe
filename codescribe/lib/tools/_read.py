# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from codescribe.lib.util import guess_media_mime, media_attach_marker
from codescribe.lib.tools._base import PathTool, schema


class ReadTool(PathTool):
    def __init__(self, root: Optional[Path] = None) -> None:
        super().__init__(
            "read",
            "Read a text file. Supports optional 1-indexed offset and line limit. "
            "By default, prefixes each returned line with a 1-indexed line number. "
            "Media files (images, PDFs, videos) are not returned as text: reading one "
            "returns a MEDIA_ATTACH marker and the harness attaches the file to the "
            "conversation as media blocks.",
            schema(
                ["path"],
                path={"type": "string", "description": "Path to the file to read."},
                offset={
                    "type": "integer",
                    "description": "1-indexed starting line number.",
                    "minimum": 1,
                },
                limit={
                    "type": "integer",
                    "description": "Maximum number of lines to read.",
                    "minimum": 1,
                },
                with_line_numbers={
                    "type": "integer",
                    "description": "Set to 1 (default) to prefix each returned line with its line number; 0 to return raw text.",
                },
            ),
            root=root,
        )

    def run(self, args: Dict[str, Any]) -> str:
        path = args.get("path")
        if not path:
            return "Error: missing required argument 'path'"
        try:
            path = self.resolve(path)
        except Exception as exc:
            return f"Error: {exc}"

        if not os.path.exists(path):
            return f"Error: file not found: {path}"
        if not os.path.isfile(path):
            return f"Error: not a file: {path}"
        if guess_media_mime(path):
            return media_attach_marker(path)

        try:
            with open(path, "r", errors="replace") as fh:
                lines = fh.readlines()
        except Exception as exc:
            return f"Error: {exc}"

        limit = args.get("limit")
        start = max(int(args.get("offset") or 1), 1) - 1
        end = len(lines) if limit is None else start + max(int(limit), 1)
        chunk = lines[start:end]
        if not chunk:
            return ""
        if not int(args.get("with_line_numbers", 1) or 0):
            return "".join(chunk)

        width = max(4, len(str(end)))
        header = f"# path: {path}\n# lines: {start + 1}-{min(end, len(lines))}\n"
        return header + "".join(
            f"{i:0{width}d}: {line}" for i, line in enumerate(chunk, start=start + 1)
        )
