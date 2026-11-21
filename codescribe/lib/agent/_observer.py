# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Run observers: console presentation kept out of the agent's control flow."""

from __future__ import annotations

import contextlib
import sys
from typing import Iterator

from alive_progress import alive_bar

from codescribe.lib.agent._output import clip, summarize_tool_output
from codescribe.lib.agent._run import RunResult, TokenUsage


class RunObserver:
    """No-op observer; override hooks to render progress."""

    @contextlib.contextmanager
    def model_call(self, iteration: int) -> Iterator[None]:
        yield

    def on_run_start(self, run_id: str, max_iterations: int) -> None:
        ...

    def on_model_response(
        self, *, iteration: int, usage: TokenUsage, reasoning: str
    ) -> None:
        ...

    def on_tool_start(self, name: str, args_preview: str) -> None:
        ...

    def on_tool_end(self, name: str, output: str) -> None:
        ...

    def on_run_end(self, result: RunResult) -> None:
        ...


def _usage_line(label: str, u: TokenUsage) -> str:
    line = f"{label}  in {u.input:,}  out {u.output:,}"
    if u.reasoning:
        line += f"  rsn {u.reasoning:,}"
    if u.cache_write or u.cache_read:
        line += f"  cache_write {u.cache_write:,}  cache_read {u.cache_read:,}"
    return line + f"  total {u.total:,}"


class ConsoleObserver(RunObserver):
    """Prints per-iteration reasoning, tool calls and token usage."""

    @contextlib.contextmanager
    def model_call(self, iteration: int) -> Iterator[None]:
        with alive_bar(
            None,
            title=f"  iter {iteration}",
            spinner="waves",
            bar=None,
            monitor=False,
            stats=False,
            elapsed=True,
            receipt=True,
        ):
            yield

    def on_model_response(
        self, *, iteration: int, usage: TokenUsage, reasoning: str
    ) -> None:
        if reasoning:
            tty = getattr(sys.stdout, "isatty", lambda: False)()
            dim, reset = ("\033[2m", "\033[0m") if tty else ("", "")
            for line in reasoning.splitlines() or [""]:
                print(f"    │ {dim}{line}{reset}")
        print(_usage_line("    usage", usage))

    def on_tool_start(self, name: str, args_preview: str) -> None:
        print(f"    ▸ {name:<5}  {args_preview:<60}", end="", flush=True)

    def on_tool_end(self, name: str, output: str) -> None:
        summary, _ = summarize_tool_output(name, output)
        first = (summary.splitlines() or ["(empty)"])[0].strip()
        print(f"  {clip(first, 70)}")

    def on_run_end(self, result: RunResult) -> None:
        print()
        print(_usage_line("  tokens", result.usage))
