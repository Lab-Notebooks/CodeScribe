# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Standalone baremetal coding agent (native tool-calling only)."""

import contextlib
import copy
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from alive_progress import alive_bar

from codescribe import lib

__all__ = [
    "Agent",
    "AgentPolicy",
    "TokenUsage",
    "ToolResult",
    "RejectedCall",
    "RunResult",
    "RunObserver",
    "ConsoleObserver",
]

_REACT_NUDGE = """\
You are a coding agent with access to tools.

Available tools (high level):
- read: read file contents
- glob: find files by pattern
- bash: run shell commands
- edit: exact text replacements in a file
- write: create/overwrite a file

Rules:
- Be concise and practical.
- Use tools whenever you need to inspect files, run commands, or change the filesystem.
- Do NOT fabricate tool outputs. If you need info, call a tool.
- Batch ALL independent reads and globs into a single turn before acting — gather everything you need first, then implement.
- Once you have the information needed, implement immediately without further exploration.
- Do not re-read files you have already read unless they were modified since your last read.
- Prefer one comprehensive edit over multiple small edits to the same file.
- IMPORTANT: Only use the tools listed here. For shell work, ONLY use the bash tool and ONLY run commands that succeed under the bash tool's safety policy. If a command is blocked, pick an allowed alternative.
- Avoid repeating identical tool calls with the same arguments unless the workspace changed (e.g., after an edit).
- Before using edit, ensure you have read the exact file region you are changing.
- When all required actions are complete, respond with the final answer immediately — do not do additional cleanup or exploration.
"""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentPolicy:
    """Tunable execution-policy knobs for a run.

    Grouped here so the loop body carries no magic numbers and policy can be
    injected (and tested) without editing Agent internals.
    """

    max_tool_calls_total: int = 120
    max_calls_per_iteration: int = 10
    max_repeated_calls: int = 2
    read_repeat_multiplier: int = (
        3  # reads get more repeats (paging / post-edit verify)
    )
    max_consecutive_error_iters: int = 3
    max_history_chars: int = 8000  # cap a single tool output kept in message history


@dataclass
class TokenUsage:
    """Normalized token accounting, accumulable with ``+``/``+=``."""

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
            input=self.input + other.input,
            output=self.output + other.output,
            reasoning=self.reasoning + other.reasoning,
            cache_write=self.cache_write + other.cache_write,
            cache_read=self.cache_read + other.cache_read,
            cache_write_5m=self.cache_write_5m + other.cache_write_5m,
            cache_write_1h=self.cache_write_1h + other.cache_write_1h,
        )

    @classmethod
    def from_raw(cls, usage: Optional[Dict[str, Any]]) -> "TokenUsage":
        """Build from a provider usage dict (OpenAI- or Anthropic-shaped)."""
        if not usage:
            return cls()
        input_tokens = usage.get("prompt_tokens")
        if input_tokens is None:
            input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens")
        if output_tokens is None:
            output_tokens = usage.get("output_tokens", 0)
        return cls(
            input=int(input_tokens or 0),
            output=int(output_tokens or 0),
            reasoning=int(usage.get("reasoning_tokens", 0) or 0),
            cache_write=int(usage.get("cache_creation_input_tokens", 0) or 0),
            cache_read=int(usage.get("cache_read_input_tokens", 0) or 0),
            cache_write_5m=int(usage.get("cache_creation_5m_input_tokens", 0) or 0),
            cache_write_1h=int(usage.get("cache_creation_1h_input_tokens", 0) or 0),
        )


@dataclass
class ToolResult:
    """One executed tool call, surfaced to callers via RunResult.

    Only real executions are recorded here (blocked-repeat and unparseable-args
    calls are not) so downstream summaries reflect actual workspace actions.
    """

    name: str
    args: Dict[str, Any]
    ok: bool
    output_preview: str  # first ~500 chars of the raw output


@dataclass
class RejectedCall:
    """A tool call the agent attempted but the harness refused to execute.

    Distinct from a ToolResult: no workspace action occurred, so these are kept
    out of the verified-action record. They are carried separately so downstream
    summaries (and the review agent) can flag claims about them as unverified.
    """

    name: str
    args: Dict[str, Any]
    reason: str  # "repeat_blocked" | "bad_json" | "iteration_skip"


@dataclass
class IterationTelemetry:
    """Normalized per-iteration telemetry for downstream loop analysis."""

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
    """Structured outcome of Agent.run().

    Replaces the old string return so callers can distinguish a real answer
    from a budget/iteration stop, and read usage + actions without re-parsing
    the event log.
    """

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
    """Per-run mutable state. Created fresh for every run() so Agent instances
    are reusable and not shared-state across concurrent runs."""

    tool_calls_total: int = 0
    call_counts: Dict[str, int] = field(default_factory=dict)
    recent: List[Dict[str, Any]] = field(default_factory=list)
    recent_errors: List[str] = field(default_factory=list)
    consecutive_error_iters: int = 0
    tool_results: List[ToolResult] = field(default_factory=list)
    rejected_calls: List[RejectedCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    iteration_telemetry: List[IterationTelemetry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Diagnostics observer (separates console presentation from control flow)
# ---------------------------------------------------------------------------


class RunObserver:
    """No-op observer base. Override to render run progress."""

    @contextlib.contextmanager
    def model_call(self, iteration: int) -> Iterator[None]:
        yield

    def on_run_start(self, run_id: str, max_iterations: int) -> None:
        ...

    def on_model_response(
        self, *, iteration: int, usage: "TokenUsage", reasoning: str
    ) -> None:
        ...

    def on_tool_start(self, name: str, args_preview: str) -> None:
        ...

    def on_tool_end(self, name: str, output: str) -> None:
        ...

    def on_run_end(self, result: "RunResult") -> None:
        ...


class ConsoleObserver(RunObserver):
    """Renders the original verbose console diagnostics."""

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

    @staticmethod
    def cache_part(usage: "TokenUsage") -> str:
        if usage.cache_write or usage.cache_read:
            return (
                f"  cache_write {usage.cache_write:,}  cache_read {usage.cache_read:,}"
            )
        return ""

    @staticmethod
    def print_reasoning_block(text: str) -> None:
        """Print reasoning lines prefixed with │, dimmed when stdout is a TTY."""
        use_ansi = getattr(sys.stdout, "isatty", lambda: False)()
        dim = "\033[2m" if use_ansi else ""
        reset = "\033[0m" if use_ansi else ""
        for line in text.splitlines() or [""]:
            print(f"    │ {dim}{line}{reset}")

    @staticmethod
    def diagnostic_result_hint(name: str, output: str) -> str:
        """One-line hint for console diagnostics."""
        summary, _ = summarize_tool_output(name, output)
        first = (summary.splitlines() or ["(empty)"])[0].strip()
        return (first[:70] + "…") if len(first) > 70 else first

    def on_model_response(
        self, *, iteration: int, usage: "TokenUsage", reasoning: str
    ) -> None:
        if reasoning:
            self.print_reasoning_block(reasoning)
        rsn_part = f"  rsn {usage.reasoning:,}" if usage.reasoning else ""
        print(
            f"    usage  in {usage.input:,}  out {usage.output:,}"
            f"{rsn_part}{self.cache_part(usage)}  total {usage.total:,}"
        )

    def on_tool_start(self, name: str, args_preview: str) -> None:
        print(f"    ▸ {name:<5}  {args_preview:<60}", end="", flush=True)

    def on_tool_end(self, name: str, output: str) -> None:
        print(f"  {self.diagnostic_result_hint(name, output)}")

    def on_run_end(self, result: "RunResult") -> None:
        u = result.usage
        rsn_part = f"  rsn {u.reasoning:,}" if u.reasoning else ""
        print()
        print(
            f"  tokens  in {u.input:,}  out {u.output:,}"
            f"{rsn_part}{self.cache_part(u)}  total {u.total:,}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_error_output(output: str) -> bool:
    return (output or "").lstrip().startswith("Error:")


def validate_schema_value(
    schema: Dict[str, Any], value: Any, path: str = "args"
) -> Optional[str]:
    expected_type = schema.get("type")

    if expected_type == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object"
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                return f"missing required argument {key!r}"
        if schema.get("additionalProperties") is False:
            extras = [k for k in value if k not in props]
            if extras:
                return f"unexpected argument(s): {', '.join(repr(k) for k in extras)}"
        for key, item in value.items():
            if key in props:
                err = validate_schema_value(props[key], item, f"{path}.{key}")
                if err:
                    return err
        return None

    if expected_type == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            return f"{path} must contain at least {min_items} item(s)"
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(value):
                err = validate_schema_value(item_schema, item, f"{path}[{idx}]")
                if err:
                    return err
        return None

    if expected_type == "string":
        if not isinstance(value, str):
            return f"{path} must be a string"
        return None

    if expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{path} must be an integer"
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            return f"{path} must be >= {minimum}"
        return None

    return None


def parse_bash_exit_code(output: str) -> Optional[int]:
    first = (output.splitlines() or [""])[0].strip()
    m = re.match(r"exit_code:\s*(-?\d+)\s*$", first)
    return int(m.group(1)) if m else None


def summarize_tool_output(name: str, output: str) -> Tuple[str, bool]:
    """Return (summary_text, attach_hint) — used only for the workspace context block."""
    out = output or ""
    out_s = out.strip()

    if is_error_output(out_s):
        msg = out_s[6:].strip()
        msg = (msg[:400] + "…") if len(msg) > 400 else msg
        return f"Error: {msg}", True

    if name == "bash":
        code = parse_bash_exit_code(out)
        lines = [
            l for l in out.splitlines()[1:] if l and l not in ("STDOUT:", "STDERR:")
        ]
        head = lines[:12]
        tail = lines[-12:] if len(lines) > 24 else []
        body = head + (["…(snip)…"] if tail else []) + tail
        body_txt = "\n".join(body).strip() or "(no output)"
        if code is not None and code != 0:
            return f"bash exit_code={code}\n{body_txt}", True
        return f"bash exit_code={code if code is not None else '?'}\n{body_txt}", False

    if not out_s:
        return "(empty)", False

    lines = out.splitlines()
    if len(lines) <= 24 and len(out) <= 1500:
        return out_s, False

    head = lines[:12]
    tail = lines[-12:]
    omitted = max(0, len(lines) - (len(head) + len(tail)))
    summary = "\n".join(head)
    if omitted:
        summary += f"\n…[truncated {omitted} lines]…\n" + "\n".join(tail)
    return summary.strip(), False


class Agent:
    """Standalone coding agent (native tools only).

    Config (model, tools, policy, observer) lives on the instance; all per-run
    mutable state lives in a RunState created inside run(), so an Agent can be
    reused across runs without state bleed.
    """

    @staticmethod
    def format_args(name: str, args: Dict[str, Any]) -> str:
        """Short human-readable preview of tool arguments."""
        if name == "bash":
            cmd = (args.get("command") or "").replace("\n", " ").strip()
            return (cmd[:60] + "…") if len(cmd) > 60 else cmd
        if name in ("read", "write"):
            return args.get("path", "")
        if name == "edit":
            path = args.get("path", "")
            n = len(args.get("edits") or [])
            return f"{path}  ({n} edit{'s' if n != 1 else ''})"
        s = json.dumps(args, ensure_ascii=False)
        return (s[:60] + "…") if len(s) > 60 else s

    @staticmethod
    def stable_json(obj: Any) -> str:
        try:
            return json.dumps(obj, sort_keys=True, ensure_ascii=False)
        except Exception:
            return json.dumps(str(obj), ensure_ascii=False)

    @classmethod
    def tool_call_key(cls, name: str, args: Dict[str, Any]) -> str:
        return f"{name}:{cls.stable_json(args)}"

    def emit(self, payload: Dict[str, Any]) -> None:
        try:
            self.logging.emit(payload)
        except Exception:
            pass

    def __init__(
        self,
        model: lib.Model,
        tools: Optional[List[lib.AgentTool]] = None,
        max_iterations: int = 20,
        show_diagnostics: bool = False,
        logging: Optional[Any] = None,
        policy: Optional[AgentPolicy] = None,
        observer: Optional[RunObserver] = None,
    ) -> None:

        allowed_types = lib.ALLOWED_MODEL_TYPES
        if not isinstance(model, allowed_types):
            allowed = ", ".join(t.__name__ for t in allowed_types)
            raise TypeError(
                f"Invalid model type: {type(model).__name__}. Expected one of: {allowed}"
            )

        if not getattr(model, "supports_native_tools", False) or not hasattr(
            model, "chat_with_tools"
        ):
            raise RuntimeError(
                f"Model {type(model).__name__} must support native tool calling (chat_with_tools)."
            )

        self.model = model
        source = tools if tools is not None else lib.DEFAULT_TOOLS
        self._tools: Dict[str, lib.AgentTool] = {t.name: copy.copy(t) for t in source}
        self.max_iterations = max_iterations
        self.show_diagnostics = show_diagnostics
        self.policy = policy if policy is not None else AgentPolicy()

        # Diagnostics presentation is decoupled from control flow via an observer.
        if observer is not None:
            self._observer = observer
        else:
            self._observer = ConsoleObserver() if show_diagnostics else RunObserver()

        # Tool log sink (JSONL/TOML recommended). Must expose .emit(dict).
        # This is the structured inspection channel; control flow does not depend on it.
        self.logging = logging if logging is not None else lib.NullToolLogSink()

    def enable_tool(self, name: str) -> None:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name!r}")
        self._tools[name].enabled = True

    def disable_tool(self, name: str) -> None:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name!r}")
        self._tools[name].enabled = False

    def enabled_tools(self) -> List[lib.AgentTool]:
        return [tool for tool in self._tools.values() if tool.enabled]

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self.enabled_tools()]

    def execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        iteration: Optional[int] = None,
        model_text: Optional[str] = None,
    ) -> str:
        t = lib.Timer()
        ok = True
        error: Optional[str] = None
        output: str

        if name not in self._tools:
            ok = False
            error = f"unknown tool {name!r}"
            output = f"Error: {error}"
        else:
            tool = self._tools[name]
            if not tool.enabled:
                ok = False
                error = f"tool {name!r} is disabled"
                output = f"Error: {error}"
            elif not isinstance(args, dict):
                ok = False
                error = f"tool {name!r} arguments must be an object"
                output = f"Error: {error}"
            else:
                err = validate_schema_value(tool.parameters, args)
                if err:
                    ok = False
                    error = err
                    output = f"Error: {err}"
                else:
                    self.emit(
                        {
                            "event": "tool_start",
                            "run_id": run_id,
                            "iteration": iteration,
                            "tool": name,
                            "args": args,
                            "model_text": model_text or None,
                        }
                    )

                    try:
                        output = tool.run(args)
                    except Exception as exc:
                        ok = False
                        error = repr(exc)
                        output = f"Error: {exc}"

                    # Detect tools that signal failure via error-string return
                    # rather than raising (e.g. "Error: file not found",
                    # "Error: blocked shell syntax detected").
                    if ok and is_error_output(output):
                        ok = False
                        error = output.lstrip()[6:].strip()

        self.emit(
            {
                "event": "tool_end",
                "run_id": run_id,
                "iteration": iteration,
                "tool": name,
                "ok": ok,
                "error": error,
                "duration_ms": round(t.ms, 3),
                "output_chars": len(output),
                "output_preview": output[:500] if output else None,
            }
        )

        return output

    def record_rejected_call(
        self,
        state: RunState,
        name: str,
        args: Dict[str, Any],
        reason: str,
        *,
        run_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> None:
        """Record a tool call that was refused before execution.

        These never touch the workspace, so they are kept out of tool_results.
        They are emitted to the log and carried on RunResult so downstream
        consumers (e.g. the review agent) can flag claims about them.
        """
        state.rejected_calls.append(
            RejectedCall(
                name=name,
                args=dict(args) if isinstance(args, dict) else {},
                reason=reason,
            )
        )
        self.emit(
            {
                "event": "tool_rejected",
                "run_id": run_id,
                "iteration": iteration,
                "tool": name,
                "args": args if isinstance(args, dict) else {},
                "reason": reason,
            }
        )

    def record_tool_result(
        self, state: RunState, name: str, args: Dict[str, Any], output: str
    ) -> None:
        summary, _ = summarize_tool_output(name, output)

        rec = {
            "tool": name,
            "args_preview": self.format_args(name, args),
            "summary": summary[:2000],
            "ok": not is_error_output(output or ""),
        }
        state.recent.append(rec)
        state.recent = state.recent[-12:]

        if not rec["ok"]:
            state.recent_errors.append(f"{name}: {summary[:400]}")
            state.recent_errors = state.recent_errors[-5:]

    def workspace_context_block(
        self, state: RunState, *, iteration: int, max_iterations: int
    ) -> str:
        tool_calls_total = int(state.tool_calls_total)
        recent = state.recent
        recent_errors = state.recent_errors

        lines: List[str] = []
        lines.append("WORKSPACE CONTEXT")
        lines.append(f"- iteration: {iteration}/{max_iterations}")
        lines.append(
            f"- tool_calls_total: {tool_calls_total}/{self.policy.max_tool_calls_total}"
        )

        if recent_errors:
            lines.append("- recent_errors:")
            for e in recent_errors[-3:]:
                lines.append(f"  - {e}")

        if recent:
            lines.append("- recent_tool_results:")
            for r in recent[-10:]:
                s = (r.get("summary") or "").replace("\n", " | ")
                s = (s[:240] + "…") if len(s) > 240 else s
                lines.append(f"  - {r['tool']}({r['args_preview']}): {s}")

        return "\n".join(lines).strip()

    @staticmethod
    def upsert_workspace_context(messages: List[Dict[str, Any]], block: str) -> None:
        """Append or update a dedicated trailing WORKSPACE CONTEXT user message.

        Keeping this out of the system prompt lets the static system message stay
        identical across every iteration, which is required for Anthropic prompt
        caching to hit. Keeping it in a dedicated trailing user message also
        avoids mutating earlier user/tool content, which helps OpenAI-compatible
        prompt caching preserve a stable prefix.
        """
        _MARKER = "WORKSPACE CONTEXT"

        if messages:
            last = messages[-1]
            if last.get("role") == "user":
                content = last.get("content", "")
                if isinstance(content, str) and content.startswith(_MARKER):
                    messages[-1] = {"role": "user", "content": block}
                    return

        messages.append({"role": "user", "content": block})

    def handle_tool_calls(
        self,
        state: RunState,
        tool_calls: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        response: Dict[str, Any],
        model_text: str,
        run_id: str,
        iteration: int,
    ) -> Optional[str]:
        """Execute one iteration's tool calls.

        Returns a stop_reason string ("tool_budget") to halt the run, or None to
        continue. Mutates state and messages in place.
        """
        if state.tool_calls_total >= self.policy.max_tool_calls_total:
            return "tool_budget"

        limited_calls = tool_calls[: self.policy.max_calls_per_iteration]
        skipped = len(tool_calls) - len(limited_calls)

        outputs: List[str] = []
        executed_calls: List[Dict[str, Any]] = []

        for call in limited_calls:
            call_name = call.get("name", "")
            call_args = call.get("arguments", {})

            # Preserve non-dict arguments for better self-correction.
            raw_args = call.get("_raw_arguments")
            raw_args_err = call.get("_raw_arguments_error")

            if not isinstance(call_args, dict):
                call_args = {}

            # Balanced repetition policy:
            # - Allow more repeats for read (often used for paging / post-edit verification)
            # - Reset repeat counts after successful edit/write (workspace changed)
            max_repeats = (
                (self.policy.max_repeated_calls * self.policy.read_repeat_multiplier)
                if call_name == "read"
                else self.policy.max_repeated_calls
            )

            key = self.tool_call_key(call_name, call_args)
            state.call_counts[key] = int(state.call_counts.get(key, 0)) + 1

            if state.call_counts[key] > max_repeats:
                hint = (
                    f"Error: repeated tool call blocked after {max_repeats} tries. "
                    "Change arguments (e.g., different offset/limit/path) or change approach."
                )
                self.record_tool_result(state, call_name, call_args, hint)
                self.record_rejected_call(
                    state,
                    call_name,
                    call_args,
                    "repeat_blocked",
                    run_id=run_id,
                    iteration=iteration,
                )
                outputs.append(hint)
                executed_calls.append(call)
                continue

            self._observer.on_tool_start(
                call_name, self.format_args(call_name, call_args)
            )

            # If the provider emitted invalid JSON arguments, surface that clearly.
            if raw_args_err:
                output = (
                    "Error: tool call arguments were not valid JSON and could not be parsed.\n"
                    f"tool={call_name!r}\n"
                    f"parse_error={raw_args_err}\n"
                    f"raw_arguments={raw_args!r}\n"
                    "Fix: emit a tool call with a JSON object matching the tool schema."
                )
                self.record_rejected_call(
                    state,
                    call_name,
                    call_args,
                    "bad_json",
                    run_id=run_id,
                    iteration=iteration,
                )
            else:
                output = self.execute_tool(
                    call_name,
                    call_args,
                    run_id=run_id,
                    iteration=iteration,
                    model_text=model_text or None,
                )
                # Count only real tool executions against the total budget, and
                # record only real executions in the structured result.
                state.tool_calls_total += 1
                state.tool_results.append(
                    ToolResult(
                        name=call_name,
                        args=dict(call_args),
                        ok=not is_error_output(output or ""),
                        output_preview=(output or "")[:500],
                    )
                )

            self.record_tool_result(state, call_name, call_args, output)

            # If workspace changed, allow repeats again (read-after-edit, re-run bash, etc.).
            if (not is_error_output(output or "")) and call_name in ("edit", "write"):
                state.call_counts = {}

            # Cap very large outputs in the message history to prevent the context
            # window from growing unboundedly across iterations. Errors and small
            # outputs are always passed in full.
            if (
                not is_error_output(output)
                and len(output) > self.policy.max_history_chars
            ):
                msg_output = (
                    output[: self.policy.max_history_chars]
                    + f"\n…[output truncated: {len(output) - self.policy.max_history_chars} chars omitted."
                    " Use read(path, offset=N) to page through the rest.]"
                )
            else:
                msg_output = output
            outputs.append(msg_output)
            executed_calls.append(call)

            self._observer.on_tool_end(call_name, output)

        if skipped > 0:
            for call in tool_calls[self.policy.max_calls_per_iteration :]:
                call_args = call.get("arguments", {})
                self.record_rejected_call(
                    state,
                    call.get("name", ""),
                    call_args,
                    "iteration_skip",
                    run_id=run_id,
                    iteration=iteration,
                )
            outputs.append(
                f"Note: {skipped} tool call(s) were skipped this iteration due to "
                f"max_tool_calls_per_iteration={self.policy.max_calls_per_iteration}."
            )
            executed_calls.append(
                {"id": "skipped_tool_calls_note", "name": "note", "arguments": {}}
            )

        messages.extend(
            self.model.format_tool_result_messages(
                executed_calls,
                outputs,
                reasoning_blocks=response.get("reasoning_blocks"),
            )
        )

        # Stuck-loop detection: count consecutive iterations where every tool call
        # failed; nudge the agent to emit a BLOCKED answer.
        real_outputs = [o for o in outputs if not o.startswith("Note:")]
        if real_outputs and all(is_error_output(o) for o in real_outputs):
            state.consecutive_error_iters += 1
        else:
            state.consecutive_error_iters = 0

        if state.consecutive_error_iters >= self.policy.max_consecutive_error_iters:
            state.consecutive_error_iters = 0
            _stuck = (
                "Every tool call for several consecutive iterations has failed. "
                "Stop attempting tool calls and output a BLOCKED: section "
                "listing exactly what is preventing progress and what the user must resolve."
            )
            last = messages[-1]
            if isinstance(last.get("content"), str):
                messages[-1] = dict(last, content=last["content"] + "\n\n" + _stuck)
            else:
                messages.append({"role": "user", "content": _stuck})

        return None

    def run(
        self,
        task: str,
        system: str = "",
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> RunResult:
        """Run the agent on a task and return a structured RunResult."""

        messages: List[Dict[str, Any]] = []

        sys_parts: List[str] = []
        if system:
            sys_parts.append(system.strip())
        sys_parts.append(_REACT_NUDGE.strip())
        messages.append(
            {"role": "system", "content": "\n\n".join(p for p in sys_parts if p)}
        )

        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": task})

        state = RunState()
        empty_reply_nudged = False

        run_id = lib.new_run_id()
        self.emit(
            {
                "event": "run_start",
                "run_id": run_id,
                "mode": "native_tools",
                "max_iterations": self.max_iterations,
            }
        )
        self._observer.on_run_start(run_id, self.max_iterations)

        stop_reason = "max_iterations"
        final_text: Optional[str] = None
        iters_done = 0

        for iteration in range(self.max_iterations):
            iters_done = iteration + 1
            self.emit(
                {"event": "iteration_start", "run_id": run_id, "iteration": iters_done}
            )

            # Refresh compact workspace context each iteration (token efficient grounding).
            self.upsert_workspace_context(
                messages,
                self.workspace_context_block(
                    state, iteration=iters_done, max_iterations=self.max_iterations
                ),
            )

            model_timer = lib.Timer()
            with self._observer.model_call(iters_done):
                response = self.model.chat_with_tools(messages, self.tool_schemas())

            text = (response.get("text") or "").strip()
            tool_calls = response.get("tool_calls") or []
            reasoning = (response.get("reasoning") or "").strip()
            usage_raw = response.get("usage") or getattr(self.model, "last_usage", None)
            iter_usage = TokenUsage.from_raw(usage_raw)
            state.usage += iter_usage

            self._observer.on_model_response(
                iteration=iters_done, usage=iter_usage, reasoning=reasoning
            )
            self.emit(
                {
                    "event": "model_response",
                    "run_id": run_id,
                    "iteration": iters_done,
                    "duration_ms": round(model_timer.ms, 3),
                    "text_chars": len(text),
                    "tool_calls": len(tool_calls),
                    "usage": usage_raw,
                    "model_text": text[:500] if text else None,
                    "model_reasoning": reasoning[:500] if reasoning else None,
                }
            )
            state.iteration_telemetry.append(
                IterationTelemetry(
                    iteration=iters_done,
                    input_tokens=iter_usage.input,
                    output_tokens=iter_usage.output,
                    reasoning_tokens=iter_usage.reasoning,
                    cache_write_tokens=iter_usage.cache_write,
                    cache_read_tokens=iter_usage.cache_read,
                    tool_calls_requested=len(tool_calls),
                    had_final_text=bool(text),
                )
            )

            # ReAct rule: if tool calls exist, execute them even if some text is also present.
            if tool_calls:
                stop = self.handle_tool_calls(
                    state, tool_calls, messages, response, text, run_id, iters_done
                )
                if stop is not None:
                    stop_reason = stop
                    final_text = None
                    break
                self.emit(
                    {
                        "event": "iteration_end",
                        "run_id": run_id,
                        "iteration": iters_done,
                        "stop_reason": "tool_calls",
                    }
                )
                continue

            if text:
                final_text = text
                stop_reason = "final_text"
                break

            if not empty_reply_nudged:
                empty_reply_nudged = True
                if any(
                    "JSON" in e or "not valid JSON" in e for e in state.recent_errors
                ):
                    nudge = (
                        "Your last tool call could not be parsed because the arguments "
                        "were not valid JSON. Please re-emit the tool call with a properly "
                        "formed JSON object matching the tool schema."
                    )
                else:
                    nudge = (
                        "Reply with either tool_calls (to take an action) "
                        "or a final answer as text."
                    )
                last = messages[-1]
                if isinstance(last.get("content"), str):
                    messages[-1] = dict(last, content=last["content"] + "\n\n" + nudge)
                else:
                    messages.append({"role": "user", "content": nudge})
                continue

        result = RunResult(
            final_text=final_text,
            stop_reason=stop_reason,
            usage=state.usage,
            iterations=iters_done,
            tool_results=state.tool_results,
            rejected_calls=state.rejected_calls,
            iteration_telemetry=state.iteration_telemetry,
        )

        self._observer.on_run_end(result)
        self.emit(
            {
                "event": "run_end",
                "run_id": run_id,
                "iteration": iters_done,
                "stop_reason": stop_reason,
                "total_input_tokens": state.usage.input,
                "total_output_tokens": state.usage.output,
                "total_reasoning_tokens": state.usage.reasoning,
                "total_cache_creation_tokens": state.usage.cache_write,
                "total_cache_read_tokens": state.usage.cache_read,
            }
        )

        return result
