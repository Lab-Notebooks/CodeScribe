# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

from __future__ import annotations

import os, importlib, json

from typing import Any, Dict, List, Optional, Union

__all__ = [
    "OpenAICompModel",
    "AnthropicModel",
    "ALLOWED_MODEL_TYPES",
    "Model",
    "set_neural_model",
]


class OpenAICompModel:
    outputs = 1

    def __init__(
        self,
        model: str,
        profile: str = "oaic",
        reasoning: bool = False,
    ) -> None:
        openai = importlib.import_module("openai")

        self.model = model
        self.profile = profile
        # Max tokens the model may generate per reply (output tokens only).
        # Read per instance, not at class level, so CODESCRIBE_MAX_TOKENS
        # applies whenever the model is constructed.
        self.max_tokens = int(os.getenv("CODESCRIBE_MAX_TOKENS", "32768"))
        self.reasoning_enabled = reasoning or _env_flag(
            "CODESCRIBE_MODEL_REASONING", False
        )
        # Reasoning, when enabled, always runs at high effort.
        self.reasoning_effort: Optional[str] = (
            "high" if self.reasoning_enabled else None
        )

        if profile == "openai":
            self.apikey = os.getenv("OPENAI_API_KEY")
            if not self.apikey:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.baseurl = None
            self.provider = None
            self.pipeline = openai.OpenAI(api_key=self.apikey)
        elif profile == "oaic":
            self.baseurl = os.getenv("OPENAI_COMP_BASEURL")
            if not self.baseurl:
                raise ValueError("OPENAI_COMP_BASEURL environment variable is not set")

            self.provider = os.getenv("OPENAI_COMP_PROVIDER")
            if not self.provider:
                raise ValueError("OPENAI_COMP_PROVIDER environment variable is not set")

            self.apikey = os.getenv("OPENAI_COMP_APIKEY")
            if not self.apikey:
                raise ValueError("OPENAI_COMP_APIKEY environment variable is not set")

            self.pipeline = openai.OpenAI(api_key=self.apikey, base_url=self.baseurl)
        else:
            raise ValueError(
                f"Unknown OpenAI profile '{profile}'. Use 'openai' or 'oaic'."
            )

        self.last_usage = None
        self._openai = openai

    @property
    def supports_native_tools(self) -> bool:
        return True

    def chat(self, chat_template: List[Dict[str, str]]) -> str:
        kwargs = self._request_kwargs(messages=chat_template)
        normalized = self._stream_chat_completion(kwargs)
        if normalized is None:
            response = self.pipeline.chat.completions.create(**kwargs)
            self.last_usage = _normalize_openai_usage(getattr(response, "usage", None))
            normalized = self._normalize_message(
                response.choices[0].message, self.last_usage
            )
        return "\n\n".join(
            part for part in (normalized["reasoning"], normalized["text"]) if part
        )

    def chat_with_tools(
        self, chat_template: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        kwargs = self._request_kwargs(messages=chat_template, tools=tools)
        normalized = self._stream_chat_completion(kwargs)
        if normalized is not None:
            return normalized

        response = self.pipeline.chat.completions.create(**kwargs)
        self.last_usage = _normalize_openai_usage(getattr(response, "usage", None))
        return self._normalize_message(response.choices[0].message, self.last_usage)

    def format_tool_result_messages(
        self,
        tool_calls: List[Dict[str, Any]],
        outputs: List[str],
        reasoning_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        assistant_tool_calls = []
        for call in tool_calls:
            assistant_tool_calls.append(
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    },
                }
            )

        assistant_content = None
        if reasoning_blocks:
            assistant_content = (
                "\n\n".join(
                    block.get("text", "")
                    for block in reasoning_blocks
                    if block.get("text")
                )
                or None
            )

        messages: List[Dict[str, Any]] = [
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": assistant_tool_calls,
            }
        ]
        for call, output in zip(tool_calls, outputs):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": output,
                }
            )
        return messages

    def _request_kwargs(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        chat_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role in {"system", "user", "assistant", "tool"}:
                chat_messages.append(msg)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": self.max_tokens,
            "n": self.outputs,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def _stream_chat_completion(
        self, kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        # include_usage adds a final chunk carrying token counts. Strict
        # OpenAI-compatible servers reject the field, so retry without it
        # before giving up on streaming.
        streaming = {**kwargs, "stream": True}
        try:
            stream = self.pipeline.chat.completions.create(
                **streaming, stream_options={"include_usage": True}
            )
        except Exception:
            try:
                stream = self.pipeline.chat.completions.create(**streaming)
            except Exception:
                return None

        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        usage = None
        tool_calls: Dict[int, Dict[str, Any]] = {}

        try:
            for chunk in stream:
                usage = getattr(chunk, "usage", None) or usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                content = getattr(delta, "content", None)
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type")
                            btext = block.get("text") or block.get("summary")
                        else:
                            btype = getattr(block, "type", None)
                            btext = getattr(block, "text", None) or getattr(
                                block, "summary", None
                            )
                        if btype == "text" and btext:
                            text_parts.append(btext)
                        elif btype in ("reasoning", "summary_text") and btext:
                            reasoning_parts.append(btext)

                reasoning = getattr(delta, "reasoning", None)
                if reasoning is not None:
                    if isinstance(reasoning, str):
                        reasoning_parts.append(reasoning)
                    else:
                        summary = getattr(reasoning, "summary", None)
                        if isinstance(summary, str):
                            reasoning_parts.append(summary)
                        elif isinstance(summary, list):
                            for item in summary:
                                if isinstance(item, str):
                                    reasoning_parts.append(item)
                                else:
                                    text = getattr(item, "text", None)
                                    if text:
                                        reasoning_parts.append(text)

                for call in getattr(delta, "tool_calls", None) or []:
                    index = getattr(call, "index", None)
                    if index is None:
                        index = len(tool_calls)
                    entry = tool_calls.setdefault(
                        index,
                        {"id": None, "name": None, "arguments": ""},
                    )
                    if getattr(call, "id", None):
                        entry["id"] = call.id
                    function = getattr(call, "function", None)
                    if function is not None:
                        if getattr(function, "name", None):
                            entry["name"] = function.name
                        arguments = getattr(function, "arguments", None)
                        if arguments:
                            entry["arguments"] += arguments
        except Exception:
            return None

        normalized_tool_calls: List[Dict[str, Any]] = []
        for idx in sorted(tool_calls):
            call = tool_calls[idx]
            raw_args_str = call["arguments"] or "{}"
            raw_args_err: str | None = None
            try:
                arguments = json.loads(raw_args_str)
            except Exception as exc:
                arguments = {}
                raw_args_err = f"{type(exc).__name__}: {exc}"

            item: Dict[str, Any] = {
                "id": call["id"],
                "name": call["name"],
                "arguments": arguments,
            }
            if raw_args_err is not None:
                item["_raw_arguments"] = raw_args_str
                item["_raw_arguments_error"] = raw_args_err
            normalized_tool_calls.append(item)

        reasoning_text = "\n\n".join(
            p
            for i, p in enumerate((p.strip() for p in reasoning_parts if p))
            if p and p not in reasoning_parts[:i]
        )
        normalized_usage = _normalize_openai_usage(usage)
        self.last_usage = normalized_usage
        return {
            "text": "".join(text_parts),
            "tool_calls": normalized_tool_calls,
            "usage": normalized_usage,
            "reasoning": reasoning_text,
            "reasoning_blocks": (
                [{"type": "reasoning", "text": reasoning_text}]
                if reasoning_text
                else []
            ),
        }

    def _normalize_message(self, message: Any, usage: Any = None) -> Dict[str, Any]:
        parts: List[str] = []
        reasoning = getattr(message, "reasoning", None)
        if reasoning is not None:
            if isinstance(reasoning, str):
                parts.append(reasoning)
            else:
                summary = getattr(reasoning, "summary", None)
                if isinstance(summary, str):
                    parts.append(summary)
                elif isinstance(summary, list):
                    for item in summary:
                        if isinstance(item, str):
                            parts.append(item)
                        else:
                            text = getattr(item, "text", None)
                            if text:
                                parts.append(text)

        content = getattr(message, "content", None)
        text = content if isinstance(content, str) else ""
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    btext = block.get("text") or block.get("summary")
                else:
                    btype = getattr(block, "type", None)
                    btext = getattr(block, "text", None) or getattr(
                        block, "summary", None
                    )
                if btype == "text" and btext:
                    text = f"{text}{btext}"
                elif btype in ("reasoning", "summary_text") and btext:
                    parts.append(btext)

        reasoning_text = "\n\n".join(
            p
            for i, p in enumerate((p.strip() for p in parts if p))
            if p and p not in parts[:i]
        )

        tool_calls = []
        for call in getattr(message, "tool_calls", []) or []:
            raw_args = call.function.arguments or "{}"
            raw_args_str = raw_args if isinstance(raw_args, str) else str(raw_args)
            raw_args_err: str | None = None
            try:
                arguments = json.loads(raw_args_str)
            except Exception as exc:
                arguments = {}
                raw_args_err = f"{type(exc).__name__}: {exc}"

            item: Dict[str, Any] = {
                "id": call.id,
                "name": call.function.name,
                "arguments": arguments,
            }
            if raw_args_err is not None:
                item["_raw_arguments"] = raw_args_str
                item["_raw_arguments_error"] = raw_args_err
            tool_calls.append(item)

        return {
            "text": text,
            "tool_calls": tool_calls,
            "usage": _normalize_openai_usage(usage),
            "reasoning": reasoning_text,
            "reasoning_blocks": (
                [{"type": "reasoning", "text": reasoning_text}]
                if reasoning_text
                else []
            ),
        }

    def __repr__(self) -> str:
        return (
            f"OpenAICompModel(model='{self.model}', profile='{self.profile}', "
            f"outputs={self.outputs}, max_tokens={self.max_tokens})"
        )


class AnthropicModel:
    def __init__(
        self,
        model: str,
        reasoning: bool = False,
    ) -> None:
        anthropic = importlib.import_module("anthropic")

        self.apikey = os.getenv("ANTHROPIC_API_KEY")
        if not self.apikey:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        self.base_url = os.getenv("ANTHROPIC_BASE_URL")
        self.reasoning_enabled = reasoning or _env_flag(
            "CODESCRIBE_MODEL_REASONING", False
        )
        self.thinking = (
            {"type": "adaptive", "display": "summarized"}
            if self.reasoning_enabled
            else None
        )

        client_kwargs = {"api_key": self.apikey}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        # The 1-hour TTL is requested per-block via cache_control.ttl in
        # _request_kwargs; this header alone does nothing and is kept only for
        # older gateways behind ANTHROPIC_BASE_URL.
        client_kwargs["default_headers"] = {
            "anthropic-beta": "extended-cache-ttl-2025-04-11"
        }

        self.client = anthropic.Anthropic(**client_kwargs)
        self.model = model
        self.max_tokens = int(os.getenv("CODESCRIBE_MAX_TOKENS", "32768"))
        self.last_usage = None

    @property
    def supports_native_tools(self) -> bool:
        return True

    def chat(self, chat_template: List[Dict[str, str]]) -> str:
        kwargs = self._request_kwargs(chat_template)

        # Newer anthropic-sdk-python versions require streaming for long requests
        # (server-side enforcement for operations that may exceed ~10 minutes).
        # Always attempt streaming, but fall back to a plain create() call if
        # the SDK/provider doesn't support it.
        try:
            stream = self.client.messages.stream(**kwargs)
        except Exception:
            stream = None
        if stream is not None:
            text_parts: List[str] = []
            start_usage = None
            delta_output_tokens = 0
            with stream as s:
                for event in s:
                    et = getattr(event, "type", None)
                    if et == "message_start":
                        msg = getattr(event, "message", None)
                        if msg is not None:
                            start_usage = getattr(msg, "usage", None)
                    elif et == "message_delta":
                        du = getattr(event, "usage", None)
                        if du is not None:
                            delta_output_tokens = int(
                                getattr(du, "output_tokens", 0) or 0
                            )
                    elif et == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if getattr(delta, "type", None) == "text_delta":
                            text_parts.append(getattr(delta, "text", "") or "")

            self.last_usage = _merge_stream_usage(start_usage, delta_output_tokens)
            return "".join(text_parts)

        response = self.client.messages.create(**kwargs)
        self.last_usage = _normalize_anthropic_usage(getattr(response, "usage", None))
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def chat_with_tools(
        self, chat_template: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        kwargs = self._request_kwargs(chat_template, tools)

        # Always attempt streaming; accumulate events and normalize into the
        # same {text, tool_calls, usage} shape as non-streaming. Falls back to
        # a plain create() call if the SDK/provider doesn't support it.
        try:
            stream = self.client.messages.stream(**kwargs)
        except Exception:
            stream = None
        if stream is not None:
            text_parts: List[str] = []
            final_message = None
            start_usage = None
            delta_output_tokens = 0

            with stream as s:
                for event in s:
                    et = getattr(event, "type", None)
                    if et == "message_start":
                        msg = getattr(event, "message", None)
                        if msg is not None:
                            start_usage = getattr(msg, "usage", None)
                    elif et == "message_delta":
                        du = getattr(event, "usage", None)
                        if du is not None:
                            delta_output_tokens = int(
                                getattr(du, "output_tokens", 0) or 0
                            )
                    elif et == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if getattr(delta, "type", None) == "text_delta":
                            text_parts.append(getattr(delta, "text", "") or "")
                try:
                    final_message = s.get_final_message()
                except Exception:
                    pass

            # Use event-captured usage (reliable even with extended thinking);
            # get_final_message() is still attempted for content extraction.
            usage = _merge_stream_usage(start_usage, delta_output_tokens)
            self.last_usage = usage

            response = final_message
            if response is None:
                return {
                    "text": "".join(text_parts),
                    "tool_calls": [],
                    "usage": usage,
                }

            normalized = _normalize_anthropic_tool_response(response, usage)
            if not normalized.get("text"):
                normalized["text"] = "".join(text_parts)
            return normalized

        response = self.client.messages.create(**kwargs)
        usage = _normalize_anthropic_usage(getattr(response, "usage", None))
        self.last_usage = usage
        return _normalize_anthropic_tool_response(response, usage)

    def format_tool_result_messages(
        self,
        tool_calls: List[Dict[str, Any]],
        outputs: List[str],
        reasoning_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        assistant_content: List[Dict[str, Any]] = []
        # Reasoning (thinking) blocks must be echoed back verbatim before tool_use blocks.
        for tb in reasoning_blocks or []:
            assistant_content.append(tb)
        for call in tool_calls:
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["arguments"],
                }
            )

        user_content = []
        for call, output in zip(tool_calls, outputs):
            user_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": output,
                }
            )

        return [
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": user_content},
        ]

    def _request_kwargs(
        self,
        chat_template: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        system_parts: List[str] = []
        messages: List[Dict[str, Any]] = []
        for msg in chat_template:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                messages.append(msg)

        # Rolling breakpoint on the second-to-last user turn: already frozen, so
        # turn N reads what turn N-1 wrote. Kept at the default 5m TTL since the
        # entry is superseded next turn (tools/system below are written at 1h,
        # and longer TTLs must precede shorter ones in the prefix).
        if len(messages) >= 2:
            messages = list(messages)
            n_user = 0
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    n_user += 1
                    if n_user == 2:
                        m = messages[i]
                        c = m.get("content", "")
                        if isinstance(c, str):
                            messages[i] = dict(
                                m,
                                content=[
                                    {
                                        "type": "text",
                                        "text": c,
                                        "cache_control": {"type": "ephemeral"},
                                    }
                                ],
                            )
                        elif isinstance(c, list) and c:
                            last = c[-1]
                            if isinstance(last, dict) and "cache_control" not in last:
                                messages[i] = dict(
                                    m,
                                    content=c[:-1]
                                    + [
                                        {**last, "cache_control": {"type": "ephemeral"}}
                                    ],
                                )
                        break

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system_parts:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_parts[0],
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
                *[{"type": "text", "text": p} for p in system_parts[1:]],
            ]
        if tools is not None:
            anthropic_tools = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"]["parameters"],
                }
                for tool in tools
            ]
            if anthropic_tools:
                anthropic_tools[-1] = {
                    **anthropic_tools[-1],
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            kwargs["tools"] = anthropic_tools
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        return kwargs

    def __repr__(self) -> str:
        return f"AnthropicModel(model='{self.model}')"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes")


def _attr(obj: Any, *names: str) -> Any:
    """First non-None attribute (or key, for dicts) among `names`, else None."""
    if obj is None:
        return None
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _normalize_openai_usage(usage: Any) -> Any:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage

    normalized = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            normalized[key] = value

    # o1/o3/o4-mini models nest reasoning_tokens under completion_tokens_details.
    details = getattr(usage, "completion_tokens_details", None)
    rt = getattr(details, "reasoning_tokens", None) if details is not None else None
    if rt is None:
        # Some OpenAI-compatible providers expose it at the top level.
        rt = getattr(usage, "reasoning_tokens", None)
    if rt is not None:
        normalized["reasoning_tokens"] = int(rt)

    # Prompt caching counters: Chat Completions nests these under
    # prompt_tokens_details, other surfaces under input_tokens_details or at the
    # top level. Only newer models bill writes separately and report them.
    details = _attr(usage, "prompt_tokens_details", "input_tokens_details")
    ct = _attr(details, "cached_tokens")
    if ct is None:
        ct = _attr(usage, "cached_tokens")
    if ct is not None:
        normalized["cache_read_input_tokens"] = int(ct)

    cw = _attr(details, "cache_write_tokens", "cache_creation_tokens")
    if cw is None:
        cw = _attr(usage, "cache_write_tokens", "cache_creation_input_tokens")
    if cw is not None:
        normalized["cache_creation_input_tokens"] = int(cw)

    # Anthropic reports input_tokens as the tokens *after* the last breakpoint
    # (total = input + read + creation) but OpenAI's prompt_tokens includes
    # them, so subtract to stop TokenUsage.total double-counting cache hits.
    cached_total = int(ct or 0) + int(cw or 0)
    if cached_total:
        for key in ("prompt_tokens", "input_tokens"):
            if key in normalized:
                normalized[key] = max(0, int(normalized[key]) - cached_total)

    if not normalized and hasattr(usage, "model_dump"):
        return usage.model_dump()
    if not normalized and hasattr(usage, "dict"):
        return usage.dict()
    return normalized or None


def _merge_stream_usage(
    start_usage: Any, delta_output_tokens: int
) -> Optional[Dict[str, Any]]:
    """Build a normalized usage dict from Anthropic streaming event data.

    message_start carries input_tokens + cache fields; message_delta carries
    output_tokens. Combining them avoids depending on get_final_message(), which
    can fail when extended thinking blocks are present.
    """
    result: Dict[str, Any] = {}
    if start_usage is not None:
        for key in (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            val = getattr(start_usage, key, None)
            if isinstance(val, int):
                result[key] = val
        result.update(_anthropic_cache_creation_split(start_usage))
    if delta_output_tokens:
        result["output_tokens"] = delta_output_tokens
    return result or None


def _anthropic_cache_creation_split(usage: Any) -> Dict[str, int]:
    creation = _attr(usage, "cache_creation")
    if creation is None:
        return {}
    split: Dict[str, int] = {}
    for src, dst in (
        ("ephemeral_5m_input_tokens", "cache_creation_5m_input_tokens"),
        ("ephemeral_1h_input_tokens", "cache_creation_1h_input_tokens"),
    ):
        val = _attr(creation, src)
        if isinstance(val, int):
            split[dst] = val
    return split


def _normalize_anthropic_usage(usage: Any) -> Any:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage

    normalized = {}
    for src, dst in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("cache_creation_input_tokens", "cache_creation_input_tokens"),
        ("cache_read_input_tokens", "cache_read_input_tokens"),
        # Anthropic exposes thinking token usage when extended thinking is active.
        ("thinking_tokens", "reasoning_tokens"),
    ):
        value = getattr(usage, src, None)
        if value is not None:
            normalized[dst] = value

    normalized.update(_anthropic_cache_creation_split(usage))

    if not normalized and hasattr(usage, "model_dump"):
        return usage.model_dump()
    if not normalized and hasattr(usage, "dict"):
        return usage.dict()
    return normalized or None


def _normalize_anthropic_tool_response(
    response: Any, usage: Any = None
) -> Dict[str, Any]:
    texts = []
    tool_calls = []
    reasoning_parts: List[str] = []
    # Raw dicts must keep Anthropic's wire format {"type":"thinking","thinking":"..."}
    # so they can be echoed back verbatim in the next assistant turn.
    reasoning_blocks: List[Dict[str, Any]] = []
    for block in response.content:
        if block.type == "thinking":
            t = block.thinking or ""
            reasoning_parts.append(t)
            rb: Dict[str, Any] = {"type": "thinking", "thinking": t}
            sig = getattr(block, "signature", None)
            if sig:
                rb["signature"] = sig
            reasoning_blocks.append(rb)
        elif block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "name": block.name,
                    "arguments": dict(block.input or {}),
                }
            )
    return {
        "text": "\n".join(t for t in texts if t).strip(),
        "tool_calls": tool_calls,
        "usage": usage,
        "reasoning": "\n\n".join(t for t in reasoning_parts if t).strip(),
        "reasoning_blocks": reasoning_blocks,
    }


ALLOWED_MODEL_TYPES = (OpenAICompModel, AnthropicModel)
Model = Union[OpenAICompModel, AnthropicModel]


def set_neural_model(model: str, reasoning: bool = False) -> Model:
    """Instantiate and return the appropriate LLM based on the model string."""
    if model.lower().startswith("openai-"):
        return OpenAICompModel(
            model[len("openai-") :], profile="openai", reasoning=reasoning
        )

    if model.lower().startswith("anthropic-"):
        return AnthropicModel(model[len("anthropic-") :], reasoning=reasoning)

    if model.lower().startswith("oaic-"):
        return OpenAICompModel(
            model[len("oaic-") :], profile="oaic", reasoning=reasoning
        )

    raise ValueError(
        f"Unknown model '{model}'. Use a recognized prefix: "
        "openai-, anthropic-, or oaic-."
    )
