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
    # Provider "max_output_tokens" (Responses API) is the maximum number of
    # tokens the model may generate for the reply (i.e., output tokens,
    # reasoning included). Match Anthropic default to allow equally large
    # reasoning / planning replies.
    max_tokens = int(os.getenv("CODESCRIBE_MAX_TOKENS", "32768"))

    def __init__(
        self,
        model: str,
        profile: str = "oaic",
        reasoning: bool = False,
    ) -> None:
        openai = importlib.import_module("openai")

        self.model = model
        self.profile = profile
        # Reasoning, when enabled, always runs at high effort with a summary.
        self.reasoning_effort: Optional[str] = "high" if reasoning else None
        self.reasoning_summary: Optional[str] = "auto" if reasoning else None

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

    @property
    def supports_native_tools(self) -> bool:
        return True

    def chat(self, chat_template: List[Dict[str, str]]) -> str:
        response = self.pipeline.responses.create(
            **self._request_kwargs(messages=chat_template)
        )
        self.last_usage = _normalize_responses_usage(getattr(response, "usage", None))
        normalized = self._normalize_response(response, self.last_usage)
        return "\n\n".join(
            part for part in (normalized["reasoning"], normalized["text"]) if part
        )

    def chat_with_tools(
        self, chat_template: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        response = self.pipeline.responses.create(
            **self._request_kwargs(
                messages=chat_template, tools=self._to_responses_tools(tools)
            )
        )
        self.last_usage = _normalize_responses_usage(getattr(response, "usage", None))
        return self._normalize_response(response, self.last_usage)

    def format_tool_result_messages(
        self,
        tool_calls: List[Dict[str, Any]],
        outputs: List[str],
        reasoning_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        # Native Responses API input items. Reasoning items must precede the
        # function_call items they were produced alongside, and must be
        # echoed back verbatim (encrypted_content is opaque) rather than
        # reconstructed, mirroring how AnthropicModel replays `thinking`
        # blocks with their `signature` untouched.
        items: List[Dict[str, Any]] = []
        for block in reasoning_blocks or []:
            items.append(block)
        for call in tool_calls:
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                }
            )
        for call, output in zip(tool_calls, outputs):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call["id"],
                    "output": output,
                }
            )
        return items

    def _to_responses_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if tools is None:
            return None
        return [
            {
                "type": "function",
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "parameters": tool["function"]["parameters"],
                "strict": False,
            }
            for tool in tools
        ]

    def _request_kwargs(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        instructions_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                instructions_parts.append(msg.get("content", ""))
            else:
                input_items.append(msg)

        if len(input_items) >= 2:
            input_items = list(input_items)
            n_user = 0
            for i in range(len(input_items) - 1, -1, -1):
                if input_items[i].get("role") == "user":
                    n_user += 1
                    if n_user == 2:
                        m = input_items[i]
                        c = m.get("content", "")
                        if isinstance(c, str):
                            input_items[i] = dict(
                                m,
                                content=[
                                    {
                                        "type": "input_text",
                                        "text": c,
                                        "prompt_cache_breakpoint": {
                                            "mode": "explicit"
                                        },
                                    }
                                ],
                            )
                        elif isinstance(c, list) and c:
                            last = c[-1]
                            if (
                                isinstance(last, dict)
                                and "prompt_cache_breakpoint" not in last
                            ):
                                input_items[i] = dict(
                                    m,
                                    content=c[:-1]
                                    + [
                                        {
                                            **last,
                                            "prompt_cache_breakpoint": {
                                                "mode": "explicit"
                                            },
                                        }
                                    ],
                                )
                        break

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": self.max_tokens,
            "store": False,
        }
        if instructions_parts:
            kwargs["instructions"] = "\n\n".join(p for p in instructions_parts if p)
        if tools is not None:
            kwargs["tools"] = tools
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": self.reasoning_summary,
            }
            kwargs["include"] = ["reasoning.encrypted_content"]
        kwargs["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
        return kwargs

    def _normalize_response(self, response: Any, usage: Any = None) -> Dict[str, Any]:
        text = getattr(response, "output_text", None)
        if not isinstance(text, str):
            text = ""

        tool_calls: List[Dict[str, Any]] = []
        reasoning_blocks: List[Dict[str, Any]] = []
        reasoning_parts: List[str] = []

        for item in getattr(response, "output", None) or []:
            itype = getattr(item, "type", None)
            if itype == "message" and not text:
                for block in getattr(item, "content", None) or []:
                    if getattr(block, "type", None) == "output_text":
                        text += getattr(block, "text", "") or ""
            elif itype == "function_call":
                raw_args = getattr(item, "arguments", None) or "{}"
                raw_args_str = raw_args if isinstance(raw_args, str) else str(raw_args)
                raw_args_err: str | None = None
                try:
                    arguments = json.loads(raw_args_str)
                except Exception as exc:
                    arguments = {}
                    raw_args_err = f"{type(exc).__name__}: {exc}"

                call_item: Dict[str, Any] = {
                    "id": getattr(item, "call_id", None),
                    "name": getattr(item, "name", None),
                    "arguments": arguments,
                }
                if raw_args_err is not None:
                    call_item["_raw_arguments"] = raw_args_str
                    call_item["_raw_arguments_error"] = raw_args_err
                tool_calls.append(call_item)
            elif itype == "reasoning":
                dump = None
                if hasattr(item, "model_dump"):
                    dump = item.model_dump(exclude_none=True)
                elif isinstance(item, dict):
                    dump = dict(item)
                if dump is not None:
                    reasoning_blocks.append(dump)
                for summary in getattr(item, "summary", None) or []:
                    summary_text = getattr(summary, "text", None)
                    if summary_text:
                        reasoning_parts.append(summary_text)

        reasoning_text = "\n\n".join(p for p in reasoning_parts if p)

        return {
            "text": text,
            "tool_calls": tool_calls,
            "usage": usage,
            "reasoning": reasoning_text,
            "reasoning_blocks": reasoning_blocks,
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
        # Request 1-hour cache TTL instead of the default 5 minutes so the
        # system prompt and tool schemas stay warm across loop boundaries.
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
                    "cache_control": {"type": "ephemeral"},
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
                    "cache_control": {"type": "ephemeral"},
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


def _normalize_responses_usage(usage: Any) -> Any:
    """Normalize a Responses API ``ResponseUsage`` into the shared usage dict
    shape ``TokenUsage.from_raw`` (see ``_agent.py``) expects."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage

    normalized: Dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            normalized[key] = value

    input_details = getattr(usage, "input_tokens_details", None)
    cached = getattr(input_details, "cached_tokens", None) if input_details else None
    if cached is not None:
        normalized["cache_read_input_tokens"] = int(cached)
    cache_write = (
        getattr(input_details, "cache_write_tokens", None) if input_details else None
    )
    if cache_write is not None:
        normalized["cache_creation_input_tokens"] = int(cache_write)

    output_details = getattr(usage, "output_tokens_details", None)
    rt = getattr(output_details, "reasoning_tokens", None) if output_details else None
    if rt is not None:
        normalized["reasoning_tokens"] = int(rt)

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
    if delta_output_tokens:
        result["output_tokens"] = delta_output_tokens
    return result or None


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
