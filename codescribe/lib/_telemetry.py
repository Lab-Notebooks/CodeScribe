# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Loop telemetry persistence helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

from codescribe import lib

__all__ = [
    "ensure_loop_metadata_dir",
    "write_loop_phase_metadata",
    "write_loop_manifest",
]


def ensure_loop_metadata_dir(loop_dir: Path) -> Path:
    path = loop_dir / "metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def write_loop_phase_metadata(
    metadata_dir: Path,
    *,
    run_id: str,
    loop_index: int,
    phase: str,
    model: str,
    task_file: str,
    workdir: str,
    stop_reason: str,
    final_text_present: bool,
    usage: Any,
    iterations: int,
    tool_results: Sequence[Any],
    rejected_calls: Sequence[Any],
    duration_s: float,
) -> Path:
    tool_rows = [_plain(item) for item in tool_results]
    rejected_rows = [_plain(item) for item in rejected_calls]
    tool_errors = sum(1 for row in tool_rows if not bool(row.get("ok", False)))
    phase_doc: Dict[str, Any] = {
        "run_id": run_id,
        "loop_index": int(loop_index),
        "phase": phase,
        "model": model,
        "task_file": task_file,
        "workdir": workdir,
        "stop_reason": stop_reason,
        "final_text_present": bool(final_text_present),
        "iterations": int(iterations),
        "duration_s": round(float(duration_s), 3),
        "usage": _plain(usage),
        "tool_calls": {
            "executed": len(tool_rows),
            "rejected": len(rejected_rows),
            "errors": tool_errors,
            "ok": len(tool_rows) - tool_errors,
        },
        "tools": tool_rows,
        "rejected_calls": rejected_rows,
        "created_at": lib.iso_utc_now(),
    }
    out = metadata_dir / f"loop_{loop_index:03d}_{phase}.toml"
    lib.atomic_write_toml(out, phase_doc)
    return out


def write_loop_manifest(
    metadata_dir: Path,
    *,
    run_doc: Dict[str, Any],
    phase_files: Sequence[Path],
) -> Path:
    manifest = {
        "run": dict(run_doc),
        "phase_files": [p.name for p in sorted(phase_files)],
        "updated_at": lib.iso_utc_now(),
    }
    out = metadata_dir / "manifest.toml"
    lib.atomic_write_toml(out, manifest)
    return out
