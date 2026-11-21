# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Tool base classes and path-bounding helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def schema(required: List[str], **properties: Dict[str, Any]) -> Dict[str, Any]:
    """JSON schema for a tool's argument object."""
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def ensure_within_workdir(path: Path, workdir: Path) -> Path:
    """Resolve `path` (against the cwd) and require it to live under `workdir`."""
    path, workdir = path.resolve(), workdir.resolve()
    try:
        path.relative_to(workdir)
    except ValueError:
        raise ValueError(f"Path {path} is outside working directory {workdir}")
    return path


class AgentTool:
    """A named, schema-described action the agent can call."""

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
        """Resolve `target` (relative to `root`) and require it to stay under `root`."""
        root = root.resolve()
        candidate = Path(target)
        candidate = (
            candidate if candidate.is_absolute() else root / candidate
        ).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise ValueError(f"Path escapes working directory: {target}")
        return candidate

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


class PathTool(AgentTool):
    """A tool acting on files, optionally bounded to `root` and barred from `protected_paths`."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        root: Optional[Path] = None,
        protected_paths: Optional[set] = None,
    ) -> None:
        if root is not None:
            description += " Access is restricted to the working directory tree."
        super().__init__(name, description, parameters)
        self.root = Path(root).resolve() if root is not None else None
        self.protected_paths = set(protected_paths or ())

    def resolve(self, path: str) -> str:
        """`path` bounded to root (raises ValueError if it escapes)."""
        if self.root is None:
            return path
        return str(self.resolve_within_root(self.root, path))

    def write_error(self, path: str, verb: str) -> Optional[str]:
        """Error if `path` is git metadata or protected, else None."""
        if ".git" in Path(path).parts:
            return f"Error: {verb} files under .git/ is not allowed"
        try:
            protected = Path(path).resolve() in self.protected_paths
        except Exception:
            protected = False
        if protected:
            return f"Error: {verb} this file is not allowed"
        return None
