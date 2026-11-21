# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Validate tool arguments and condense tool outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple


def is_error_output(output: str) -> bool:
    return (output or "").lstrip().startswith("Error:")


def clip(text: str, limit: int) -> str:
    return (text[:limit] + "…") if len(text) > limit else text


def validate_schema_value(
    schema: Dict[str, Any], value: Any, path: str = "args"
) -> Optional[str]:
    """First mismatch between `value` and a JSON-schema subset, else None."""
    expected = schema.get("type")

    if expected == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object"
        props = schema.get("properties", {})
        missing = next((k for k in schema.get("required", []) if k not in value), None)
        if missing is not None:
            return f"missing required argument {missing!r}"
        extras = [k for k in value if k not in props]
        if schema.get("additionalProperties") is False and extras:
            return f"unexpected argument(s): {', '.join(repr(k) for k in extras)}"
        for key, item in value.items():
            err = key in props and validate_schema_value(
                props[key], item, f"{path}.{key}"
            )
            if err:
                return err
        return None

    if expected == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            return f"{path} must contain at least {min_items} item(s)"
        for idx, item in enumerate(value if schema.get("items") else []):
            err = validate_schema_value(schema["items"], item, f"{path}[{idx}]")
            if err:
                return err
        return None

    if expected == "string" and not isinstance(value, str):
        return f"{path} must be a string"

    if expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{path} must be an integer"
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            return f"{path} must be >= {minimum}"

    return None


def format_args(name: str, args: Dict[str, Any]) -> str:
    """Short preview of tool arguments."""
    if name == "bash":
        return clip((args.get("command") or "").replace("\n", " ").strip(), 60)
    if name in ("read", "write"):
        return args.get("path", "")
    if name == "edit":
        n = len(args.get("edits") or [])
        return f"{args.get('path', '')}  ({n} edit{'s' if n != 1 else ''})"
    return clip(json.dumps(args, ensure_ascii=False), 60)


def parse_bash_exit_code(output: str) -> Optional[int]:
    first = (output.splitlines() or [""])[0].strip()
    m = re.match(r"exit_code:\s*(-?\d+)\s*$", first)
    return int(m.group(1)) if m else None


def _head_tail(lines: list, keep: int = 12) -> list:
    if len(lines) <= 2 * keep:
        return lines
    return lines[:keep] + ["…(snip)…"] + lines[-keep:]


def summarize_tool_output(name: str, output: str) -> Tuple[str, bool]:
    """(summary, is_failure) of a tool output for the workspace context block."""
    out = output or ""
    out_s = out.strip()

    if is_error_output(out_s):
        return f"Error: {clip(out_s[6:].strip(), 400)}", True

    if name == "bash":
        code = parse_bash_exit_code(out)
        lines = [
            l for l in out.splitlines()[1:] if l and l not in ("STDOUT:", "STDERR:")
        ]
        body = "\n".join(_head_tail(lines)).strip() or "(no output)"
        return f"bash exit_code={'?' if code is None else code}\n{body}", code not in (
            None,
            0,
        )

    if not out_s:
        return "(empty)", False

    lines = out.splitlines()
    if len(lines) <= 24 and len(out) <= 1500:
        return out_s, False
    summary = "\n".join(lines[:12])
    omitted = len(lines) - 24
    if omitted > 0:
        summary += f"\n…[truncated {omitted} lines]…\n" + "\n".join(lines[-12:])
    return summary.strip(), False
