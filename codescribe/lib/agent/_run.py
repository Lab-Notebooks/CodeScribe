# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Run policy, accounting and results."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AgentPolicy:
    """Execution limits for one run."""

    max_tool_calls_total: int = 120
    max_calls_per_iteration: int = 10
    max_repeated_calls: int = 2
    read_repeat_multiplier: int = 3  # reads page and re-verify, so allow more repeats
    max_consecutive_error_iters: int = 3
    max_history_chars: int = 8000  # cap on one tool output kept in message history


@dataclass
class TokenUsage:
    """Normalized token accounting; supports `+`."""

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_write: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.cache_write + self.cache_read

    @property
    def cache_write_unattributed(self) -> int:
        return max(0, self.cache_write - self.cache_write_5m - self.cache_write_1h)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in fields(self)
            }
        )

    @classmethod
    def from_raw(cls, usage: Optional[Dict[str, Any]]) -> "TokenUsage":
        """From a normalized provider usage dict (OpenAI- or Anthropic-shaped)."""
        usage = usage or {}

        def get(*keys: str) -> int:
            value = next((usage[k] for k in keys if usage.get(k) is not None), 0)
            return int(value or 0)

        return cls(
            input=get("prompt_tokens", "input_tokens"),
            output=get("completion_tokens", "output_tokens"),
            reasoning=get("reasoning_tokens"),
            cache_write=get("cache_creation_input_tokens"),
            cache_read=get("cache_read_input_tokens"),
            cache_write_5m=get("cache_creation_5m_input_tokens"),
            cache_write_1h=get("cache_creation_1h_input_tokens"),
        )


@dataclass
class ToolResult:
    """A tool call that actually executed."""

    name: str
    args: Dict[str, Any]
    ok: bool
    output_preview: str  # first ~500 chars


@dataclass
class RejectedCall:
    """A tool call the harness refused; it never touched the workspace."""

    name: str
    args: Dict[str, Any]
    reason: str  # "repeat_blocked" | "bad_json" | "iteration_skip"


@dataclass
class IterationTelemetry:
    iteration: int
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    tool_calls_requested: int = 0
    had_final_text: bool = False


@dataclass
class RunResult:
    """Outcome of Agent.run()."""

    final_text: Optional[str]
    stop_reason: str  # "final_text" | "max_iterations" | "tool_budget"
    usage: TokenUsage
    iterations: int
    tool_results: List[ToolResult] = field(default_factory=list)
    rejected_calls: List[RejectedCall] = field(default_factory=list)
    iteration_telemetry: List[IterationTelemetry] = field(default_factory=list)

    def __str__(self) -> str:
        if self.final_text is not None:
            return self.final_text
        if self.stop_reason == "tool_budget":
            return "[Agent stopped: tool-call budget reached without a final answer]"
        return f"[Agent stopped: {self.stop_reason} reached without a final answer]"


@dataclass
class RunState:
    """Mutable state of one run; created fresh by Agent.run()."""

    tool_calls_total: int = 0
    call_counts: Dict[str, int] = field(default_factory=dict)
    recent: List[Dict[str, Any]] = field(default_factory=list)
    recent_errors: List[str] = field(default_factory=list)
    consecutive_error_iters: int = 0
    retried_413: bool = False
    tool_results: List[ToolResult] = field(default_factory=list)
    rejected_calls: List[RejectedCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    iteration_telemetry: List[IterationTelemetry] = field(default_factory=list)
