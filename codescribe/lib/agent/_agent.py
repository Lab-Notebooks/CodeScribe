# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Tool-calling agent loop (native tool calls only)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from codescribe.lib.util import (
    NullToolLogSink,
    Timer,
    file_parts,
    new_run_id,
    parse_media_marker,
    strip_inline_media,
)
from codescribe.lib.models import Model
from codescribe.lib.tools import AgentTool, make_tools
from codescribe.lib.agent._observer import ConsoleObserver, RunObserver
from codescribe.lib.agent._output import (
    format_args,
    is_error_output,
    summarize_tool_output,
    validate_schema_value,
)
from codescribe.lib.agent._run import (
    AgentPolicy,
    IterationTelemetry,
    RejectedCall,
    RunResult,
    RunState,
    TokenUsage,
    ToolResult,
)

SYSTEM_RULES = """\
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
- Media files (images, PDFs, videos) cannot be read as text: a read on one returns a MEDIA_ATTACH: marker and the file is attached to the conversation as media — use what is attached, do not re-read it.
- Prefer one comprehensive edit over multiple small edits to the same file.
- IMPORTANT: Only use the tools listed here. For shell work, ONLY use the bash tool and ONLY run commands that succeed under the bash tool's safety policy. If a command is blocked, pick an allowed alternative.
- Avoid repeating identical tool calls with the same arguments unless the workspace changed (e.g., after an edit).
- Before using edit, ensure you have read the exact file region you are changing.
- When all required actions are complete, respond with the final answer immediately — do not do additional cleanup or exploration.
"""

WORKSPACE_MARKER = "WORKSPACE CONTEXT"

NOTE_STUCK = (
    "Every tool call for several consecutive iterations has failed. "
    "Stop attempting tool calls and output a BLOCKED: section "
    "listing exactly what is preventing progress and what the user must resolve."
)
NOTE_BAD_JSON = (
    "Your last tool call could not be parsed because the arguments "
    "were not valid JSON. Please re-emit the tool call with a properly "
    "formed JSON object matching the tool schema."
)
NOTE_EMPTY = (
    "Reply with either tool_calls (to take an action) or a final answer as text."
)
NOTE_413 = (
    "The API gateway rejected the request with 413 (Request Entity Too Large): "
    "attached media files are inlined as base64 and the body exceeded the limit. "
    "All previously attached media has been stripped from the request. If you "
    "still need a file, re-attach at most one small file."
)


def call_key(name: str, args: Dict[str, Any]) -> str:
    try:
        return f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
    except Exception:
        return f"{name}:{json.dumps(str(args), ensure_ascii=False)}"


def append_note(messages: List[Dict[str, Any]], note: str) -> None:
    """Append `note` to a trailing text user message, or add a new one."""
    last = messages[-1]
    if isinstance(last.get("content"), str):
        messages[-1] = dict(last, content=last["content"] + "\n\n" + note)
    else:
        messages.append({"role": "user", "content": note})


class Agent:
    """Drives a model through tool calls until it answers or hits a limit.

    Configuration lives on the instance; per-run state lives in a RunState, so
    one Agent can serve many runs.
    """

    def __init__(
        self,
        model: Model,
        tools: Optional[List[AgentTool]] = None,
        max_iterations: int = 20,
        show_diagnostics: bool = False,
        logging: Optional[Any] = None,
        policy: Optional[AgentPolicy] = None,
        observer: Optional[RunObserver] = None,
    ) -> None:
        if not isinstance(model, Model):
            raise TypeError(
                f"Invalid model type: {type(model).__name__}. Expected a Model."
            )

        if tools is None:
            tools = make_tools(Path.cwd().resolve())
        self.model = model
        self._tools: Dict[str, AgentTool] = {t.name: copy.copy(t) for t in tools}
        self.max_iterations = max_iterations
        self.show_diagnostics = show_diagnostics
        self.policy = policy or AgentPolicy()
        self._observer = observer or (
            ConsoleObserver() if show_diagnostics else RunObserver()
        )
        # Structured event log (TOML); control flow never depends on it.
        self.logging = logging if logging is not None else NullToolLogSink()

    # -- tools ---------------------------------------------------------------

    def enable_tool(self, name: str, enabled: bool = True) -> None:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name!r}")
        self._tools[name].enabled = enabled

    def disable_tool(self, name: str) -> None:
        self.enable_tool(name, False)

    def enabled_tools(self) -> List[AgentTool]:
        return [t for t in self._tools.values() if t.enabled]

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return [t.to_openai_tool() for t in self.enabled_tools()]

    def emit(self, payload: Dict[str, Any]) -> None:
        try:
            self.logging.emit(payload)
        except Exception:
            pass

    # -- run -----------------------------------------------------------------

    def run(
        self,
        task: str,
        system: str = "",
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> RunResult:
        sys_prompt = "\n\n".join(p for p in (system.strip(), SYSTEM_RULES.strip()) if p)
        messages = [{"role": "system", "content": sys_prompt}, *(chat_history or [])]
        messages.append({"role": "user", "content": task})

        state = RunState()
        run_id = new_run_id()
        ids = {"run_id": run_id}
        self.emit(
            {
                "event": "run_start",
                **ids,
                "mode": "native_tools",
                "max_iterations": self.max_iterations,
            }
        )
        self._observer.on_run_start(run_id, self.max_iterations)

        stop_reason, final_text, iteration = "max_iterations", None, 0
        nudged = False
        for iteration in range(1, self.max_iterations + 1):
            ids = {"run_id": run_id, "iteration": iteration}
            self.emit({"event": "iteration_start", **ids})
            self.upsert_workspace_context(
                messages, self.workspace_context(state, iteration)
            )

            response = self.call_model(state, messages, ids)
            if response is None:
                continue  # request was shrunk after a 413; retry

            if response["tool_calls"]:
                stop = self.handle_tool_calls(state, messages, response, ids)
                if stop is not None:
                    stop_reason = stop
                    break
                self.emit(
                    {"event": "iteration_end", **ids, "stop_reason": "tool_calls"}
                )
                continue

            if response["text"]:
                final_text, stop_reason = response["text"], "final_text"
                break

            if not nudged:
                nudged = True
                bad_json = any("JSON" in e for e in state.recent_errors)
                append_note(messages, NOTE_BAD_JSON if bad_json else NOTE_EMPTY)

        result = RunResult(
            final_text=final_text,
            stop_reason=stop_reason,
            usage=state.usage,
            iterations=iteration,
            tool_results=state.tool_results,
            rejected_calls=state.rejected_calls,
            iteration_telemetry=state.iteration_telemetry,
        )
        self._observer.on_run_end(result)
        u = state.usage
        self.emit(
            {
                "event": "run_end",
                "run_id": run_id,
                "iteration": iteration,
                "stop_reason": stop_reason,
                "total_input_tokens": u.input,
                "total_output_tokens": u.output,
                "total_reasoning_tokens": u.reasoning,
                "total_cache_creation_tokens": u.cache_write,
                "total_cache_read_tokens": u.cache_read,
            }
        )
        return result

    def call_model(
        self, state: RunState, messages: List[Dict[str, Any]], ids: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """One model turn, with usage recorded. None if a 413 was absorbed."""
        timer = Timer()
        with self._observer.model_call(ids["iteration"]):
            try:
                response = self.model.chat_with_tools(messages, self.tool_schemas())
            except Exception as exc:
                # A gateway body-size limit: drop inlined media and retry once.
                if getattr(exc, "status_code", None) != 413 or state.retried_413:
                    raise
                state.retried_413 = True
                strip_inline_media(messages)
                messages.append({"role": "user", "content": NOTE_413})
                return None

        response["text"] = (response.get("text") or "").strip()
        response["tool_calls"] = response.get("tool_calls") or []
        reasoning = (response.get("reasoning") or "").strip()
        usage_raw = response.get("usage") or self.model.last_usage
        usage = TokenUsage.from_raw(usage_raw)
        state.usage += usage

        self._observer.on_model_response(
            iteration=ids["iteration"], usage=usage, reasoning=reasoning
        )
        self.emit(
            {
                "event": "model_response",
                **ids,
                "duration_ms": round(timer.ms, 3),
                "text_chars": len(response["text"]),
                "tool_calls": len(response["tool_calls"]),
                "usage": usage_raw,
                "model_text": response["text"][:500] or None,
                "model_reasoning": reasoning[:500] or None,
            }
        )
        state.iteration_telemetry.append(
            IterationTelemetry(
                iteration=ids["iteration"],
                input_tokens=usage.input,
                output_tokens=usage.output,
                reasoning_tokens=usage.reasoning,
                cache_write_tokens=usage.cache_write,
                cache_read_tokens=usage.cache_read,
                tool_calls_requested=len(response["tool_calls"]),
                had_final_text=bool(response["text"]),
            )
        )
        return response

    # -- tool calls ----------------------------------------------------------

    def handle_tool_calls(
        self,
        state: RunState,
        messages: List[Dict[str, Any]],
        response: Dict[str, Any],
        ids: Dict[str, Any],
    ) -> Optional[str]:
        """Run one turn's tool calls and append the results. Returns a stop reason or None."""
        if state.tool_calls_total >= self.policy.max_tool_calls_total:
            return "tool_budget"

        limit = self.policy.max_calls_per_iteration
        calls = response["tool_calls"][:limit]
        outputs = [self.run_call(state, call, response["text"], ids) for call in calls]

        skipped = response["tool_calls"][limit:]
        if skipped:
            for call in skipped:
                self.reject(
                    state,
                    call.get("name", ""),
                    call.get("arguments", {}),
                    "iteration_skip",
                    ids,
                )
            outputs.append(
                f"Note: {len(skipped)} tool call(s) were skipped this iteration due to "
                f"max_tool_calls_per_iteration={limit}."
            )
            calls.append(
                {"id": "skipped_tool_calls_note", "name": "note", "arguments": {}}
            )

        messages.extend(
            self.model.format_tool_result_messages(
                calls, outputs, reasoning_blocks=response.get("reasoning_blocks")
            )
        )
        self.attach_media(messages, outputs)
        self.check_stuck(state, messages, outputs)
        return None

    def run_call(
        self,
        state: RunState,
        call: Dict[str, Any],
        model_text: str,
        ids: Dict[str, Any],
    ) -> str:
        """Execute (or refuse) one tool call; returns the output kept in history."""
        name = call.get("name", "")
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            args = {}

        # Repeats are capped; reads get more (paging, post-edit verification).
        key = call_key(name, args)
        state.call_counts[key] = state.call_counts.get(key, 0) + 1
        max_repeats = self.policy.max_repeated_calls
        if name == "read":
            max_repeats *= self.policy.read_repeat_multiplier
        if state.call_counts[key] > max_repeats:
            hint = (
                f"Error: repeated tool call blocked after {max_repeats} tries. "
                "Change arguments (e.g., different offset/limit/path) or change approach."
            )
            self.record_recent(state, name, args, hint)
            self.reject(state, name, args, "repeat_blocked", ids)
            return hint

        self._observer.on_tool_start(name, format_args(name, args))
        if call.get("_raw_arguments_error"):
            output = (
                "Error: tool call arguments were not valid JSON and could not be parsed.\n"
                f"tool={name!r}\n"
                f"parse_error={call['_raw_arguments_error']}\n"
                f"raw_arguments={call.get('_raw_arguments')!r}\n"
                "Fix: emit a tool call with a JSON object matching the tool schema."
            )
            self.reject(state, name, args, "bad_json", ids)
        else:
            output = self.execute_tool(name, args, model_text=model_text or None, **ids)
            state.tool_calls_total += 1
            state.tool_results.append(
                ToolResult(name, dict(args), not is_error_output(output), output[:500])
            )

        self.record_recent(state, name, args, output)
        if name in ("edit", "write") and not is_error_output(output):
            state.call_counts = {}  # workspace changed: repeats are meaningful again
        self._observer.on_tool_end(name, output)
        return self.truncate(output)

    def execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        iteration: Optional[int] = None,
        model_text: Optional[str] = None,
    ) -> str:
        """Validate and run a tool; always returns text ("Error: ..." on failure)."""
        timer = Timer()
        ids = {"run_id": run_id, "iteration": iteration, "tool": name}
        tool = self._tools.get(name)
        if tool is None:
            error = f"unknown tool {name!r}"
        elif not tool.enabled:
            error = f"tool {name!r} is disabled"
        elif not isinstance(args, dict):
            error = f"tool {name!r} arguments must be an object"
        else:
            error = validate_schema_value(tool.parameters, args)

        ok = error is None
        if not ok:
            output = f"Error: {error}"
        else:
            self.emit(
                {"event": "tool_start", **ids, "args": args, "model_text": model_text}
            )
            try:
                output = tool.run(args)
            except Exception as exc:
                ok, error, output = False, repr(exc), f"Error: {exc}"
            if ok and is_error_output(output):
                ok, error = False, output.lstrip()[6:].strip()

        self.emit(
            {
                "event": "tool_end",
                **ids,
                "ok": ok,
                "error": error,
                "duration_ms": round(timer.ms, 3),
                "output_chars": len(output),
                "output_preview": output[:500] or None,
            }
        )
        return output

    def truncate(self, output: str) -> str:
        """Cap large successful outputs so history does not grow without bound."""
        cap = self.policy.max_history_chars
        if is_error_output(output) or len(output) <= cap:
            return output
        return (
            output[:cap] + f"\n…[output truncated: {len(output) - cap} chars omitted."
            " Use read(path, offset=N) to page through the rest.]"
        )

    def attach_media(self, messages: List[Dict[str, Any]], outputs: List[str]) -> None:
        """Turn MEDIA_ATTACH read results into content blocks on a new user turn.

        Only the newest attachment stays inline; older ones are stripped so the
        request body does not grow every turn.
        """
        for output in outputs:
            path = parse_media_marker(output)
            if not path:
                continue
            try:
                blocks = file_parts(path, self.model.api)
            except Exception as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error: could not attach media {path}: {exc}",
                    }
                )
                continue
            strip_inline_media(messages)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"[Attached media file: {path}]"},
                        *blocks,
                    ],
                }
            )

    def check_stuck(
        self, state: RunState, messages: List[Dict[str, Any]], outputs: List[str]
    ) -> None:
        """Nudge toward a BLOCKED answer after repeated all-error iterations."""
        real = [o for o in outputs if not o.startswith("Note:")]
        if real and all(is_error_output(o) for o in real):
            state.consecutive_error_iters += 1
        else:
            state.consecutive_error_iters = 0
        if state.consecutive_error_iters >= self.policy.max_consecutive_error_iters:
            state.consecutive_error_iters = 0
            append_note(messages, NOTE_STUCK)

    # -- bookkeeping ---------------------------------------------------------

    def reject(
        self, state: RunState, name: str, args: Any, reason: str, ids: Dict[str, Any]
    ) -> None:
        """Record a call the harness refused (kept apart from executed results)."""
        args = dict(args) if isinstance(args, dict) else {}
        state.rejected_calls.append(RejectedCall(name, args, reason))
        self.emit(
            {
                "event": "tool_rejected",
                **ids,
                "tool": name,
                "args": args,
                "reason": reason,
            }
        )

    @staticmethod
    def record_recent(
        state: RunState, name: str, args: Dict[str, Any], output: str
    ) -> None:
        summary, _ = summarize_tool_output(name, output)
        ok = not is_error_output(output)
        state.recent.append(
            {
                "tool": name,
                "args_preview": format_args(name, args),
                "summary": summary[:2000],
                "ok": ok,
            }
        )
        del state.recent[:-12]
        if not ok:
            state.recent_errors.append(f"{name}: {summary[:400]}")
            del state.recent_errors[:-5]

    def workspace_context(self, state: RunState, iteration: int) -> str:
        lines = [
            WORKSPACE_MARKER,
            f"- iteration: {iteration}/{self.max_iterations}",
            f"- tool_calls_total: {state.tool_calls_total}/{self.policy.max_tool_calls_total}",
        ]
        if state.recent_errors:
            lines.append("- recent_errors:")
            lines += [f"  - {e}" for e in state.recent_errors[-3:]]
        if state.recent:
            lines.append("- recent_tool_results:")
            for r in state.recent[-10:]:
                s = r["summary"].replace("\n", " | ")
                s = (s[:240] + "…") if len(s) > 240 else s
                lines.append(f"  - {r['tool']}({r['args_preview']}): {s}")
        return "\n".join(lines)

    @staticmethod
    def upsert_workspace_context(messages: List[Dict[str, Any]], block: str) -> None:
        """Replace or append the trailing WORKSPACE CONTEXT user message.

        Kept out of the system prompt (and out of earlier turns) so the cached
        prompt prefix stays identical across iterations.
        """
        last = messages[-1] if messages else {}
        content = last.get("content")
        if (
            last.get("role") == "user"
            and isinstance(content, str)
            and content.startswith(WORKSPACE_MARKER)
        ):
            messages[-1] = {"role": "user", "content": block}
        else:
            messages.append({"role": "user", "content": block})
