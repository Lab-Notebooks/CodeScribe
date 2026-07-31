# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Tool implementations for the standalone baremetal coding agent.

This module is intentionally separate from `_agent.py` to keep the core agent loop
focused on orchestration, while tool implementations (filesystem + shell) live
here.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "AgentTool",
    "ReadTool",
    "BashTool",
    "EditTool",
    "WriteTool",
    "GlobTool",
    "make_tools",
    "make_readonly_tools",
    "DEFAULT_TOOLS",
]


class AgentTool:
    """Base tool for the standalone coding agent."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.enabled = enabled

    @staticmethod
    def resolve_within_root(root: Path, target: str) -> Path:
        root = root.resolve()
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        try:
            candidate.relative_to(root)
        except ValueError:
            raise ValueError(f"Path escapes working directory: {target}")

        return candidate

    @staticmethod
    def is_git_internal(path: Any) -> bool:
        """True if any path component is '.git' (repo metadata/hooks/config)."""
        return ".git" in Path(path).parts

    @staticmethod
    def is_protected_path(path: Any, protected_paths: Optional[set]) -> bool:
        """True if `path` resolves to one of `protected_paths` (e.g. the task file)."""
        if not protected_paths:
            return False
        try:
            resolved = Path(path).resolve()
        except Exception:
            return False
        return resolved in protected_paths

    def run(self, args: Dict[str, Any]) -> str:
        raise NotImplementedError(f"{self.name}.run() is not implemented")

    def to_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def describe_for_prompt(self) -> str:
        return (
            f"- {self.name}: {self.description}\n"
            f"  JSON schema: {json.dumps(self.parameters, ensure_ascii=False)}"
        )


class ReadTool(AgentTool):
    def __init__(self, root: Optional[Path] = None) -> None:
        desc = (
            "Read a text file. Supports optional 1-indexed offset and line limit. "
            "By default, prefixes each returned line with a 1-indexed line number."
        )
        if root is not None:
            desc += " Access is restricted to the working directory tree."
        super().__init__(
            name="read",
            description=desc,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-indexed starting line number.",
                        "minimum": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read.",
                        "minimum": 1,
                    },
                    "with_line_numbers": {
                        "type": "integer",
                        "description": "Set to 1 (default) to prefix each returned line with its line number; 0 to return raw text.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            enabled=True,
        )
        self.root = Path(root).resolve() if root is not None else None

    def run(self, args: Dict[str, Any]) -> str:
        path = args.get("path")
        offset = args.get("offset")
        limit = args.get("limit")
        with_line_numbers = bool(int(args.get("with_line_numbers", 1) or 0))

        if not path:
            return "Error: missing required argument 'path'"

        if self.root is not None:
            try:
                path = str(self.resolve_within_root(self.root, path))
            except Exception as exc:
                return f"Error: {exc}"

        if not os.path.exists(path):
            return f"Error: file not found: {path}"
        if not os.path.isfile(path):
            return f"Error: not a file: {path}"

        try:
            with open(path, "r", errors="replace") as fh:
                lines = fh.readlines()
        except Exception as exc:
            return f"Error: {exc}"

        start = max(int(offset or 1), 1) - 1
        end = len(lines) if limit is None else start + max(int(limit), 1)
        chunk = lines[start:end]

        if not chunk:
            return ""

        if not with_line_numbers:
            return "".join(chunk)

        # Render with stable, 1-indexed line numbers.
        # Use a fixed width so columns line up and copy/paste remains unambiguous.
        width = max(4, len(str(end if end is not None else len(lines))))
        header = f"# path: {path}\n# lines: {start + 1}-{min(end, len(lines))}\n"
        numbered = "".join(
            f"{i:0{width}d}: {line}" for i, line in enumerate(chunk, start=start + 1)
        )
        return header + numbered


class GlobTool(AgentTool):
    def __init__(self, root: Optional[Path] = None) -> None:
        desc = (
            "List files matching a glob pattern (e.g. '**/*.py'). "
            "Returns newline-separated paths (relative to root when bounded)."
        )
        if root is not None:
            desc += " Access is restricted to the working directory tree."
        super().__init__(
            name="glob",
            description=desc,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (supports **).",
                    },
                    "root": {
                        "type": "string",
                        "description": "Optional root directory for the search (default: current/bounded root).",
                    },
                    "include_dirs": {
                        "type": "integer",
                        "description": "Set to 1 to include directories; 0/omitted to include only files.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "minimum": 1,
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            enabled=True,
        )
        self.root = Path(root).resolve() if root is not None else None

    def run(self, args: Dict[str, Any]) -> str:
        pattern = args.get("pattern")
        root_arg = args.get("root")
        include_dirs = bool(int(args.get("include_dirs") or 0))
        limit = int(args.get("limit") or 2000)

        if not pattern:
            return "Error: missing required argument 'pattern'"

        base: Path
        if self.root is not None:
            base = self.root
            if root_arg:
                try:
                    base = self.resolve_within_root(self.root, root_arg)
                except Exception as exc:
                    return f"Error: {exc}"
        else:
            base = Path(root_arg).resolve() if root_arg else Path.cwd().resolve()

        # Use Python's glob with recursive ** support.
        try:
            matches = glob(str(base / pattern), recursive=True)
        except Exception as exc:
            return f"Error: {exc}"

        out: List[str] = []
        for m in matches:
            p = Path(m)
            try:
                if self.root is not None:
                    # Ensure bounded matches can't escape.
                    p = self.resolve_within_root(self.root, str(p))
                else:
                    p = p.resolve()
            except Exception:
                continue

            if not include_dirs and p.is_dir():
                continue

            try:
                rel = str(p.relative_to(base))
            except Exception:
                rel = str(p)

            out.append(rel)

        out = sorted(set(out))
        if len(out) > limit:
            out = out[:limit]

        return "\n".join(out)


class BashTool(AgentTool):
    _BLOCKED_CHARS = set("|&;><`$\n\r")
    _DEFAULT_ALLOWED = {
        "ls",
        "pwd",
        "find",
        "grep",
        "head",
        "tail",
        "wc",
        "git",
        "test",
        "echo",
        "sed",
    }

    def __init__(
        self,
        cwd: Optional[Path] = None,
        bounded: bool = False,
        allowed_commands: Optional[set] = None,
        protected_paths: Optional[set] = None,
    ) -> None:
        self.cwd = Path(cwd).resolve() if cwd is not None else None
        self.bounded = bounded
        self.allowed_commands = set(allowed_commands or self._DEFAULT_ALLOWED)
        self.protected_paths = set(protected_paths or set())

        desc = "Execute a bash command in the current working directory."
        if cwd is not None:
            desc += (
                " Commands execute with the working directory set to the bounded root."
                " Only access files and paths within the working directory;"
                " do not navigate to or read from paths outside it."
            )
        if bounded:
            desc += " Potentially-dangerous shell syntax is blocked (pipes, redirects, $(), etc)."
            allowed = ", ".join(sorted(self.allowed_commands))
            desc += f" Allowed commands: {allowed}."

            caveats = []
            if any(cmd in self.allowed_commands for cmd in ("find", "bfs", "gfind")):
                caveats.append(
                    "find: -exec/-execdir/-ok/-okdir are rejected; use find only to locate paths."
                )
            if "rg" in self.allowed_commands:
                caveats.append(
                    "rg: --pre/--pre-glob are rejected."
                )
            if "sed" in self.allowed_commands:
                caveats.append(
                    "sed: runs with --sandbox enforced in bounded mode."
                )
            if "git" in self.allowed_commands:
                caveats.append(
                    "git: config subcommand and -c/-p/--paginate/--exec-path/--git-dir/"
                    "--work-tree/--namespace/--upload-pack/--receive-pack/--upload-archive "
                    "flags are rejected; ext:: and fd:: transport URLs are rejected."
                )
            if caveats:
                desc += " " + " ".join(caveats)

        super().__init__(
            name="bash",
            description=desc,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds.",
                        "minimum": 1,
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            enabled=True,
        )

    def validate_command(self, command: str) -> Optional[str]:
        if not self.bounded:
            return None

        if any(ch in command for ch in self._BLOCKED_CHARS):
            return "blocked shell syntax detected"

        # Only allow simple command forms and whitelisted binaries.
        try:
            parts = shlex.split(command)
        except Exception:
            return "could not parse command"

        if not parts:
            return "empty command"

        exe = parts[0]
        if exe not in self.allowed_commands:
            return f"command not allowed: {exe!r}"

        if exe in {"find", "bfs", "gfind"}:
            bad = next(
                (
                    arg
                    for arg in parts[1:]
                    if arg in {"-exec", "-execdir", "-ok", "-okdir", "-delete"}
                ),
                None,
            )
            if bad is not None:
                return f"disallowed {exe} flag {bad!r}: spawns an arbitrary program per match or deletes files"

        if exe == "rg":
            bad = next(
                (
                    arg
                    for arg in parts[1:]
                    if arg in {"--pre", "--pre-glob"}
                    or arg.startswith("--pre=")
                    or arg.startswith("--pre-glob=")
                ),
                None,
            )
            if bad is not None:
                return "disallowed rg flag --pre/--pre-glob: spawns an arbitrary preprocessor program"

        if exe == "git":
            if "config" in parts[1:]:
                return (
                    "disallowed git subcommand 'config': can set core.pager/"
                    "core.fsmonitor/alias.*/etc, turning a later plain git "
                    "invocation into arbitrary command execution"
                )
            git_flag_prefixes = (
                "--exec-path",
                "--git-dir",
                "--work-tree",
                "--namespace",
                "--upload-pack",
                "--receive-pack",
                "--upload-archive",
            )
            bad = next(
                (
                    arg
                    for arg in parts[1:]
                    if arg in {"-c", "-p", "--paginate"}
                    or arg.startswith(git_flag_prefixes)
                ),
                None,
            )
            if bad is not None:
                return f"disallowed git flag {bad!r}: can override config/paths or force hook execution"
            bad = next(
                (arg for arg in parts[1:] if "ext::" in arg or "fd::" in arg),
                None,
            )
            if bad is not None:
                return f"disallowed git argument {bad!r}: ext:: and fd:: transports spawn an arbitrary helper program"

        if self.cwd is not None:
            for arg in parts[1:]:
                if arg.startswith("-"):
                    continue
                try:
                    resolved = self.resolve_within_root(self.cwd, arg)
                except Exception:
                    return f"argument escapes working directory: {arg!r}"
                if resolved in self.protected_paths:
                    return f"argument targets a protected path: {arg!r}"

        return None

    def harden_command(self, command: str) -> str:
        try:
            parts = shlex.split(command)
        except Exception:
            return command
        if not parts or parts[0] != "sed" or "--sandbox" in parts:
            return command
        parts.insert(1, "--sandbox")
        return shlex.join(parts)

    def run(self, args: Dict[str, Any]) -> str:
        command = args.get("command")
        timeout = args.get("timeout")

        if not command:
            return "Error: missing required argument 'command'"

        err = self.validate_command(command)
        if err:
            return f"Error: {err}"

        if self.bounded:
            command = self.harden_command(command)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.cwd) if self.cwd is not None else None,
                capture_output=True,
                text=True,
                timeout=int(timeout) if timeout is not None else None,
            )
        except Exception as exc:
            return f"Error: {exc}"

        out = [
            f"exit_code: {proc.returncode}",
            "STDOUT:",
            proc.stdout or "",
            "STDERR:",
            proc.stderr or "",
        ]
        return "\n".join(out).rstrip() + "\n"


class EditTool(AgentTool):
    def __init__(
        self, root: Optional[Path] = None, protected_paths: Optional[set] = None
    ) -> None:
        desc = "Edit a file using exact text replacements (all oldText must match exactly once)."
        if root is not None:
            desc += " Access is restricted to the working directory tree."
        super().__init__(
            name="edit",
            description=desc,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit.",
                    },
                    "edits": {
                        "type": "array",
                        "description": "List of replacements.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {"type": "string"},
                                "newText": {"type": "string"},
                            },
                            "required": ["oldText", "newText"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                    },
                },
                "required": ["path", "edits"],
                "additionalProperties": False,
            },
            enabled=True,
        )
        self.root = Path(root).resolve() if root is not None else None
        self.protected_paths = set(protected_paths or set())

    @staticmethod
    def snippet(text: str, start: int, end: int, context: int = 80) -> str:
        """Return a compact snippet around [start:end]."""
        left = max(0, start - context)
        right = min(len(text), end + context)
        prefix = "…" if left > 0 else ""
        suffix = "…" if right < len(text) else ""
        return prefix + text[left:right] + suffix

    def run(self, args: Dict[str, Any]) -> str:
        path = args.get("path")
        edits = args.get("edits")

        if not path:
            return "Error: missing required argument 'path'"
        if not edits:
            return "Error: missing required argument 'edits'"

        if self.root is not None:
            try:
                path = str(self.resolve_within_root(self.root, path))
            except Exception as exc:
                return f"Error: {exc}"

        if self.is_git_internal(path):
            return "Error: editing files under .git/ is not allowed"
        if self.is_protected_path(path, self.protected_paths):
            return "Error: editing this file is not allowed"

        if not os.path.exists(path):
            return f"Error: file not found: {path}"
        if not os.path.isfile(path):
            return f"Error: not a file: {path}"
        try:
            with open(path, "r", errors="replace") as fh:
                content_before = fh.read()
        except Exception as exc:
            return f"Error: {exc}"

        # Validate uniqueness on the original content, and capture verification snippets.
        replacements: List[Dict[str, Any]] = []
        for idx, e in enumerate(edits):
            old = e.get("oldText")
            new = e.get("newText")
            if old is None or new is None:
                return "Error: each edit must include oldText and newText"

            count = content_before.count(old)
            if count != 1:
                return f"Error: oldText must match exactly once; found {count} matches for: {old!r}"

            start = content_before.find(old)
            end = start + len(old)
            replacements.append(
                {
                    "index": idx,
                    "match_count": count,
                    "old_len": len(old),
                    "new_len": len(new),
                    "before_snippet": self.snippet(content_before, start, end),
                    "oldText": old,
                    "newText": new,
                }
            )

        # Apply edits.
        content_after = content_before
        for r in replacements:
            content_after = content_after.replace(r["oldText"], r["newText"])

        try:
            with open(path, "w") as fh:
                fh.write(content_after)
        except Exception as exc:
            return f"Error: {exc}"

        # Final verification snippets after edit.
        for r in replacements:
            new = r["newText"]
            pos = content_after.find(new)
            if pos >= 0:
                r["after_snippet"] = self.snippet(content_after, pos, pos + len(new))
            else:
                r["after_snippet"] = "(newText not found after write)"

        report = {
            "ok": True,
            "path": path,
            "applied": len(replacements),
            "replacements": [
                {
                    "index": r["index"],
                    "match_count": r["match_count"],
                    "old_len": r["old_len"],
                    "new_len": r["new_len"],
                    "before_snippet": r["before_snippet"],
                    "after_snippet": r["after_snippet"],
                }
                for r in replacements
            ],
        }
        return json.dumps(report, ensure_ascii=False, indent=2)


class WriteTool(AgentTool):
    def __init__(
        self, root: Optional[Path] = None, protected_paths: Optional[set] = None
    ) -> None:
        desc = "Write a file (create or overwrite)."
        if root is not None:
            desc += " Access is restricted to the working directory tree."
        super().__init__(
            name="write",
            description=desc,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write."},
                    "content": {
                        "type": "string",
                        "description": "Full content to write.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            enabled=True,
        )
        self.root = Path(root).resolve() if root is not None else None
        self.protected_paths = set(protected_paths or set())

    def run(self, args: Dict[str, Any]) -> str:
        path = args.get("path")
        content = args.get("content")

        if not path:
            return "Error: missing required argument 'path'"
        if content is None:
            return "Error: missing required argument 'content'"

        if self.root is not None:
            try:
                path = str(self.resolve_within_root(self.root, path))
            except Exception as exc:
                return f"Error: {exc}"

        if self.is_git_internal(path):
            return "Error: writing files under .git/ is not allowed"
        if self.is_protected_path(path, self.protected_paths):
            return "Error: writing this file is not allowed"

        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w") as fh:
                fh.write(content)
            return f"Wrote {path} ({len(content)} bytes)"
        except Exception as exc:
            return f"Error: {exc}"


def make_tools(
    root: Path,
    bash_allow: Optional[set] = None,
    protected_paths: Optional[set] = None,
) -> List[AgentTool]:
    """Create the default bounded toolset (read/glob/bash/edit/write).

    `bash_allow` extends the default bash allowlist with additional command names.
    `protected_paths` are resolved paths (e.g. the task file) that write/edit/bash
    may never target, even though they live inside `root`.
    """
    allowed = set(BashTool._DEFAULT_ALLOWED)
    if bash_allow:
        allowed |= set(bash_allow)

    return [
        ReadTool(root=root),
        GlobTool(root=root),
        BashTool(
            cwd=root,
            bounded=True,
            allowed_commands=allowed,
            protected_paths=protected_paths,
        ),
        EditTool(root=root, protected_paths=protected_paths),
        WriteTool(root=root, protected_paths=protected_paths),
    ]


def make_readonly_tools(
    root: Path,
    bash_allow: Optional[set] = None,
    protected_paths: Optional[set] = None,
) -> List[AgentTool]:
    """Create a bounded read-only toolset (read/glob/bash only).

    `bash_allow` extends the default bash allowlist with additional command names.
    """
    allowed = set(BashTool._DEFAULT_ALLOWED)
    if bash_allow:
        allowed |= set(bash_allow)

    return [
        ReadTool(root=root),
        GlobTool(root=root),
        BashTool(
            cwd=root,
            bounded=True,
            allowed_commands=allowed,
            protected_paths=protected_paths,
        ),
    ]


DEFAULT_TOOLS: List[AgentTool] = make_tools(Path.cwd().resolve())
