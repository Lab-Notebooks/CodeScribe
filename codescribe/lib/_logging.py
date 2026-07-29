# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""_telemetry
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, List

import hashlib
import toml

from codescribe import lib

__all__ = [
    "write_archive_toml",
    "iso_utc_now",
    "new_run_id",
    "atomic_write_text",
    "atomic_write_toml",
    "read_toml",
    "append_toml_event",
    "read_toml_events",
    "ToolLogSink",
    "NullToolLogSink",
    "MultiToolLogSink",
    "ToolLogToml",
    "Timer",
]


def write_archive_toml(
    chat_entries: List[Dict[str, str]], neural_model: object
) -> None:
    """Write a chat archive TOML file under `.codescribe/chat/`.

    Folder structure: YYYY/MM/DD/timestamp_sha.toml
    """

    base_dir = Path(".codescribe") / "chat"

    now_local = datetime.now()
    timestamp = now_local.strftime("%H%M%S")
    date_path = now_local.strftime("%Y/%m/%d")
    sha = hashlib.sha1(str(now_local.timestamp()).encode()).hexdigest()[:8]

    folder = base_dir / date_path
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"{timestamp}_{sha}.toml"
    file_path = folder / filename
    file_path.touch()

    lines: List[str] = []
    for entry in chat_entries:
        role = entry["role"]
        content = entry["content"].strip()
        lines.append(f"[[chat.{role}]]")
        lines.append("content = '''")
        lines.append(content)
        lines.append("'''\n")

    metadata = [f"# {entry}" for entry in str(neural_model.__repr__()).split("\n")]
    metadata.append("\n")

    file_path.write_text("\n".join(metadata + lines))
    lib.format_seed_prompt(file_path, chat_entries)


def iso_utc_now() -> str:
    """UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    """Generate a stable, human-friendly run id."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text to *path* (best-effort on local filesystems)."""
    os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def atomic_write_toml(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write a TOML document to *path*."""
    atomic_write_text(path, toml.dumps(payload))


def read_toml(path: Path) -> Dict[str, Any]:
    """Read a TOML file, returning an empty dict if missing or unreadable."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return toml.load(fh)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# TOML event log (replaces JSONL)
# ---------------------------------------------------------------------------


def _toml_val(v: Any) -> Optional[str]:
    """Serialize a value to its TOML literal representation.

    Returns None for None (caller should skip the field).
    Complex types (dict, list) are JSON-encoded and stored as TOML strings.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(round(v, 6))
    if isinstance(v, str):
        if "\n" in v or "\r" in v:
            # Multiline literal string — no escaping needed unless it contains '''
            if "'''" not in v:
                return f"'''\n{v}'''"
            # Fall back to escaped basic string
            esc = (
                v.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
            )
            return f'"{esc}"'
        esc = v.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t")
        return f'"{esc}"'
    # dict, list → JSON-encode, then store as TOML string
    encoded = json.dumps(v, ensure_ascii=False)
    esc = encoded.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{esc}"'


def append_toml_event(path: Path, event: Mapping[str, Any]) -> None:
    """Append one event as a [[event]] block to a TOML log file."""
    os.makedirs(path.parent, exist_ok=True)
    lines = ["\n[[event]]"]
    for k, v in event.items():
        rendered = _toml_val(v)
        if rendered is not None:
            lines.append(f"{k} = {rendered}")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def read_toml_events(path: Path) -> List[Dict[str, Any]]:
    """Read all [[event]] entries from a TOML log file.

    JSON-encoded string values (written by append_toml_event for complex types
    like args/usage dicts) are automatically decoded back to Python objects.
    """
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = toml.load(fh)
    except Exception:
        return []
    events = data.get("event", [])
    # Only these fields are JSON-encoded complex types (dict/list); everything
    # else is a plain string (run_id, ts, tool names, etc.) and should not be
    # parsed — trying json.loads on every string wastes CPU for no gain.
    _JSON_FIELDS = frozenset({"args", "usage"})
    out = []
    for ev in events:
        decoded: Dict[str, Any] = {}
        for k, v in ev.items():
            if isinstance(v, str) and k in _JSON_FIELDS:
                try:
                    decoded[k] = json.loads(v)
                except (ValueError, TypeError):
                    decoded[k] = v
            else:
                decoded[k] = v
        out.append(decoded)
    return out


def _redact(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return "***"
    if isinstance(value, (list, tuple)):
        return ["***" for _ in value]
    if isinstance(value, dict):
        return {k: "***" for k in value}
    return "***"


@dataclass
class ToolLogSink:
    """Base class for tool log sinks."""

    def emit(self, event: Mapping[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class NullToolLogSink(ToolLogSink):
    def emit(self, event: Mapping[str, Any]) -> None:
        return None


class MultiToolLogSink(ToolLogSink):
    """Fan-out sink: emits each event to all child sinks."""

    def __init__(self, sinks: List[ToolLogSink]) -> None:
        self._sinks = list(sinks)

    def emit(self, event: Mapping[str, Any]) -> None:
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:
                pass


@dataclass
class ToolLogToml(ToolLogSink):
    """Append-only TOML event log sink.

    Each event is written as a [[event]] block in the TOML file.
    Complex field values (dicts, lists) are JSON-encoded as strings.

    Parameters
    ----------
    path:
        Output file path. Defaults to `.codescribe/logs/toolusage.toml`.
    redact_keys:
        Keys to redact in the event dict, at the top level.
    """

    path: Optional[str] = None
    redact_keys: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.path:
            self.path = os.path.join(".codescribe", "logs", "toolusage.toml")
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Mapping[str, Any]) -> None:
        out: Dict[str, Any] = {}
        for k, v in dict(event).items():
            if k in self.redact_keys:
                out[k] = _redact(v)
            else:
                out[k] = v
        out.setdefault("ts", iso_utc_now())
        append_toml_event(Path(self.path), out)


class Timer:
    """Simple monotonic timer."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
