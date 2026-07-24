# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Loop telemetry persistence and comparison plotting helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import toml

from codescribe import lib

__all__ = [
    "ensure_loop_metadata_dir",
    "write_loop_phase_metadata",
    "write_loop_manifest",
    "load_loop_eval_inputs",
    "plot_loop_metadata_comparison",
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


def _load_phase_file(path: Path) -> Dict[str, Any]:
    return toml.loads(path.read_text())


def _label_for_dir(path: Path) -> str:
    return path.resolve().name


def load_loop_eval_inputs(metadata_dirs: Iterable[Path]) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    for raw in metadata_dirs:
        mdir = Path(raw).resolve()
        if not mdir.exists() or not mdir.is_dir():
            raise ValueError(f"Metadata directory does not exist: {mdir}")
        manifest_path = mdir / "manifest.toml"
        if not manifest_path.exists():
            raise ValueError(f"Missing manifest.toml in metadata directory: {mdir}")
        manifest = _load_phase_file(manifest_path)
        phase_files = manifest.get("phase_files", []) or []
        phases = []
        for fname in phase_files:
            fpath = mdir / str(fname)
            if fpath.exists():
                phases.append(_load_phase_file(fpath))
        phases.sort(key=lambda row: (int(row.get("loop_index", 0)), str(row.get("phase", ""))))
        series.append({
            "label": _label_for_dir(mdir),
            "metadata_dir": str(mdir),
            "manifest": manifest,
            "phases": phases,
        })
    return series


def _phase_value(phase: Dict[str, Any], *keys: str, cast: Any = int) -> Any:
    cur: Any = phase
    for key in keys:
        if not isinstance(cur, dict):
            return cast(0)
        cur = cur.get(key, 0)
    try:
        return cast(cur or 0)
    except Exception:
        return cast(0)


def _plot_lines(ax: Any, xvals: List[int], rows: List[Dict[str, Any]], key_path: Sequence[str], title: str, ylabel: str, cast: Any = int) -> None:
    for row in rows:
        yvals = [_phase_value(phase, *key_path, cast=cast) for phase in row["phases"]]
        x = [int(phase.get("loop_index", idx + 1)) for idx, phase in enumerate(row["phases"])]
        ax.plot(x, yvals, marker="o", label=row["label"])
    ax.set_title(title)
    ax.set_xlabel("Loop")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def _plot_stacked_phase_tokens(ax: Any, rows: List[Dict[str, Any]], phase_name: str, title: str) -> None:
    labels = [row["label"] for row in rows]
    input_vals, output_vals, cache_write_vals, cache_read_vals = [], [], [], []
    for row in rows:
        total_in = total_out = total_cw = total_cr = 0
        for phase in row["phases"]:
            if str(phase.get("phase")) != phase_name:
                continue
            total_in += _phase_value(phase, "usage", "input")
            total_out += _phase_value(phase, "usage", "output")
            total_cw += _phase_value(phase, "usage", "cache_write")
            total_cr += _phase_value(phase, "usage", "cache_read")
        input_vals.append(total_in)
        output_vals.append(total_out)
        cache_write_vals.append(total_cw)
        cache_read_vals.append(total_cr)
    xpos = list(range(len(labels)))
    ax.bar(xpos, input_vals, label="input")
    ax.bar(xpos, output_vals, bottom=input_vals, label="output")
    bottom2 = [a + b for a, b in zip(input_vals, output_vals)]
    ax.bar(xpos, cache_write_vals, bottom=bottom2, label="cache_write")
    bottom3 = [a + b + c for a, b, c in zip(input_vals, output_vals, cache_write_vals)]
    ax.bar(xpos, cache_read_vals, bottom=bottom3, label="cache_read")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Tokens")


def plot_loop_metadata_comparison(
    metadata_dirs: Iterable[Path],
    *,
    output_path: Path,
) -> Path:
    rows = load_loop_eval_inputs(metadata_dirs)
    if not rows:
        raise ValueError("At least one metadata directory is required")

    plt.style.use("default")
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    _plot_lines(axes[0], [], rows, ("usage", "input"), "Input tokens per phase", "tokens")
    _plot_lines(axes[1], [], rows, ("usage", "output"), "Output tokens per phase", "tokens")
    _plot_lines(axes[2], [], rows, ("usage", "cache_write"), "Cache write tokens per phase", "tokens")
    _plot_lines(axes[3], [], rows, ("usage", "cache_read"), "Cache read tokens per phase", "tokens")
    _plot_lines(axes[4], [], rows, ("duration_s",), "Wall time per phase", "seconds", cast=float)
    axes[5].axis("off")
    _plot_stacked_phase_tokens(axes[6], rows, "author", "Author token mix")
    _plot_stacked_phase_tokens(axes[7], rows, "review", "Review token mix")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, min(4, len(labels))))
    fig.suptitle("CodeScribe loop telemetry comparison")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
