# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from codescribe.lib.tools._base import AgentTool, schema

_FIND = {"find", "bfs", "gfind"}
_GIT_FLAG_PREFIXES = (
    "--exec-path",
    "--git-dir",
    "--work-tree",
    "--namespace",
    "--upload-pack",
    "--receive-pack",
    "--upload-archive",
)


class BashTool(AgentTool):
    """Shell tool. When `bounded`, only simple allowlisted commands run."""

    DEFAULT_ALLOWED = {
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
    _BLOCKED_CHARS = set("|&;><`$\n\r")
    _PYTHON_RE = re.compile(r"^(i?python|pypy)[0-9.]*$")
    # Interpreter flags that take no argument and cannot inject code.
    _PYTHON_SAFE_FLAGS = {
        "-u",
        "-B",
        "-I",
        "-E",
        "-s",
        "-S",
        "-O",
        "-OO",
        "-q",
        "-b",
        "-bb",
        "-P",
    }

    def __init__(
        self,
        cwd: Optional[Path] = None,
        bounded: bool = False,
        allowed_commands: Optional[set] = None,
        protected_paths: Optional[set] = None,
        script_dir: Optional[Path] = None,
    ) -> None:
        self.cwd = Path(cwd).resolve() if cwd is not None else None
        self.bounded = bounded
        self.allowed_commands = set(allowed_commands or self.DEFAULT_ALLOWED)
        self.protected_paths = set(protected_paths or ())
        # New python scripts must live here, so executed python leaves a file behind.
        self.script_dir = Path(script_dir).resolve() if script_dir is not None else None
        super().__init__(
            "bash",
            self.describe(),
            schema(
                ["command"],
                command={"type": "string", "description": "Bash command to execute."},
                timeout={
                    "type": "integer",
                    "description": "Optional timeout in seconds.",
                    "minimum": 1,
                },
            ),
        )

    @classmethod
    def is_python(cls, exe: str) -> bool:
        return bool(cls._PYTHON_RE.match(exe))

    def describe(self) -> str:
        desc = "Execute a bash command in the current working directory."
        if self.cwd is not None:
            desc += (
                " Commands execute with the working directory set to the bounded root."
                " Only access files and paths within the working directory;"
                " do not navigate to or read from paths outside it."
            )
        if not self.bounded:
            return desc

        allowed = self.allowed_commands
        desc += " Potentially-dangerous shell syntax is blocked (pipes, redirects, $(), etc)."
        desc += f" Allowed commands: {', '.join(sorted(allowed))}."
        if allowed & _FIND:
            desc += " find: -exec/-execdir/-ok/-okdir are rejected; use find only to locate paths."
        if "rg" in allowed:
            desc += " rg: --pre/--pre-glob are rejected."
        if "sed" in allowed:
            desc += " sed: runs with --sandbox enforced in bounded mode."
        if "git" in allowed:
            desc += (
                " git: config subcommand and -c/-p/--paginate/--exec-path/--git-dir/"
                "--work-tree/--namespace/--upload-pack/--receive-pack/--upload-archive "
                "flags are rejected; ext:: and fd:: transport URLs are rejected."
            )
        if self.script_dir is not None and any(self.is_python(c) for c in allowed):
            rel = self.script_dir_display()
            desc += (
                f" python: may run an existing .py file anywhere in the workdir; "
                f"new scripts must be created under {rel}/ (write there first, "
                f"then run e.g. `python3 {rel}/check.py`); -c, -m, stdin "
                "and the interactive REPL are rejected."
            )
        return desc

    def script_dir_display(self) -> str:
        if self.script_dir is None:
            return ""
        if self.cwd is not None:
            try:
                return str(self.script_dir.relative_to(self.cwd))
            except ValueError:
                pass
        return str(self.script_dir)

    # -- validation: each check returns an error message or None ------------

    def check_python(self, parts: List[str]) -> Optional[str]:
        """Require `python [safe flags] <script.py> [args]`; new scripts under script_dir."""
        rel = self.script_dir_display()
        hint = (
            f"python may run an existing .py anywhere in the workdir; new scripts "
            f"must be created under {rel}/ with the write tool, then run "
            f"`{parts[0]} {rel}/<name>.py`"
        )
        script = None
        for arg in parts[1:]:
            if arg in self._PYTHON_SAFE_FLAGS:
                continue
            if arg.startswith("-"):
                return f"disallowed {parts[0]} flag {arg!r}: {hint}"
            script = arg
            break
        if script is None:
            return f"interactive/stdin python is not allowed: {hint}"
        if not script.endswith(".py"):
            return f"not a .py script: {script!r}: {hint}"
        try:
            resolved = self.resolve_within_root(self.cwd or Path.cwd(), script)
        except Exception:
            return f"argument escapes working directory: {script!r}"
        if resolved.is_file():
            return None
        if self.script_dir is not None:
            try:
                resolved.relative_to(self.script_dir)
            except ValueError:
                return f"script outside {rel}/: {script!r}: {hint}"
        return f"script not found: {script!r}: {hint}"

    @staticmethod
    def check_find(parts: List[str]) -> Optional[str]:
        bad = next(
            (
                a
                for a in parts[1:]
                if a in {"-exec", "-execdir", "-ok", "-okdir", "-delete"}
            ),
            None,
        )
        if bad is not None:
            return f"disallowed {parts[0]} flag {bad!r}: spawns an arbitrary program per match or deletes files"
        return None

    @staticmethod
    def check_rg(parts: List[str]) -> Optional[str]:
        if any(
            a in {"--pre", "--pre-glob"} or a.startswith(("--pre=", "--pre-glob="))
            for a in parts[1:]
        ):
            return "disallowed rg flag --pre/--pre-glob: spawns an arbitrary preprocessor program"
        return None

    @staticmethod
    def check_git(parts: List[str]) -> Optional[str]:
        args = parts[1:]
        if "config" in args:
            return (
                "disallowed git subcommand 'config': can set core.pager/"
                "core.fsmonitor/alias.*/etc, turning a later plain git "
                "invocation into arbitrary command execution"
            )
        bad = next(
            (
                a
                for a in args
                if a in {"-c", "-p", "--paginate"} or a.startswith(_GIT_FLAG_PREFIXES)
            ),
            None,
        )
        if bad is not None:
            return f"disallowed git flag {bad!r}: can override config/paths or force hook execution"
        bad = next((a for a in args if "ext::" in a or "fd::" in a), None)
        if bad is not None:
            return f"disallowed git argument {bad!r}: ext:: and fd:: transports spawn an arbitrary helper program"
        return None

    def check_paths(self, parts: List[str]) -> Optional[str]:
        if self.cwd is None:
            return None
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

    def validate_command(self, command: str) -> Optional[str]:
        if not self.bounded:
            return None
        if any(ch in command for ch in self._BLOCKED_CHARS):
            return "blocked shell syntax detected"
        try:
            parts = shlex.split(command)
        except Exception:
            return "could not parse command"
        if not parts:
            return "empty command"

        exe = parts[0]
        if exe not in self.allowed_commands:
            return f"command not allowed: {exe!r}"

        if self.script_dir is not None and self.is_python(exe):
            exe_check = self.check_python
        elif exe in _FIND:
            exe_check = self.check_find
        else:
            exe_check = {"rg": self.check_rg, "git": self.check_git}.get(exe)
        return (exe_check and exe_check(parts)) or self.check_paths(parts)

    @staticmethod
    def harden_command(command: str) -> str:
        """Force `sed --sandbox` so sed cannot run commands or write files."""
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
