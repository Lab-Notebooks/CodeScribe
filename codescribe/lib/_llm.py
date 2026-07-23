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
    # Provider "max_tokens" (OpenAI Responses/ChatCompletions style) is the maximum
    # number of tokens the model may generate for the reply (i.e., output tokens).
    # Default bumped to allow more verbose reasoning / planning.
    max_tokens = int(os.getenv("CODESCRIBE_MAX_TOKENS", "24576"))

    def __init__(
        self, model: str, profile: str = "oaic", reasoning: bool = False
    ) -> None:
        openai = importlib.import_module("openai")

        self.model = model
        self.profile = profile
        self.reasoning_effort: Optional[str] = "high" if reasoning else None

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
            raise ValueError(f"Unknown OpenAI profile '{profile}'. Use 'openai' or 'oaic'.")

        self.last_usage = None

    @property
    def supports_native_tools(self) -> bool:
        return True

    def chat(self, chat_template: List[Dict[str, str]]) -> str:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=chat_template,
            max_tokens=self.max_tokens,
            n=self.outputs,
        )
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = self.pipeline.chat.completions.create(**kwargs)
        self.last_usage = _normalize_openai_usage(getattr(response, "usage", None))
        return response.choices[0].message.content

    def chat_with_tools(
        self, chat_template: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=chat_template,
            tools=tools,
            max_tokens=self.max_tokens,
            n=self.outputs,
        )
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = self.pipeline.chat.completions.create(**kwargs)
        self.last_usage = _normalize_openai_usage(getattr(response, "usage", None))
        return _normalize_openai_tool_response(
            response.choices[0].message, self.last_usage
        )

    def format_tool_result_messages(
        self,
        tool_calls: List[Dict[str, Any]],
        outputs: List[str],
        reasoning_blocks: Optional[List[Dict[str, Any]]] = None,  # noqa: ARG002
    ) -> List[Dict[str, Any]]:
        # reasoning_blocks ignored — OpenAI has no equivalent exposed API.
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

        messages: List[Dict[str, Any]] = [
            {
                "role": "assistant",
                "content": None,
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
        streaming: Optional[bool] = None,
        prompt_caching: Optional[bool] = None,
        thinking: Optional[Dict[str, Any]] = None,
    ) -> None:
        anthropic = importlib.import_module("anthropic")

        self.apikey = os.getenv("ANTHROPIC_API_KEY")
        if not self.apikey:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        self.base_url = os.getenv("ANTHROPIC_BASE_URL")
        env_streaming = os.getenv("CODESCRIBE_ANTHROPIC_STREAMING", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        env_prompt_caching = os.getenv("CODESCRIBE_ANTHROPIC_CACHE", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        env_reasoning = os.getenv("CODESCRIBE_AGENT_REASONING", "").lower() in (
            "1",
            "true",
            "yes",
        )

        self.streaming = env_streaming if streaming is None else streaming
        self.prompt_caching = (
            env_prompt_caching if prompt_caching is None else prompt_caching
        )
        self.reasoning_enabled = reasoning or env_reasoning
        self.thinking = (
            thinking
            if thinking is not None
            else {"type": "adaptive", "display": "summarized"}
            if self.reasoning_enabled
            else None
        )

        client_kwargs = {"api_key": self.apikey}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.prompt_caching:
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
        system_parts, messages = anthropic_split_messages(chat_template)

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system_parts:
            kwargs["system"] = anthropic_system_param(
                system_parts, prompt_caching=self.prompt_caching
            )

        if self.streaming:
            # Newer anthropic-sdk-python versions require streaming for long requests
            # (server-side enforcement for operations that may exceed ~10 minutes).
            # Prefer streaming when enabled, but still fall back for older SDKs/providers.
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
        system_parts, messages = anthropic_split_messages(chat_template)
        messages = anthropic_apply_message_cache_breakpoint(
            messages, prompt_caching=self.prompt_caching
        )

        anthropic_tools = [_openai_tool_to_anthropic_tool(tool) for tool in tools]
        if self.prompt_caching and anthropic_tools:
            anthropic_tools[-1] = {
                **anthropic_tools[-1],
                "cache_control": {"type": "ephemeral"},
            }

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "tools": anthropic_tools,
        }
        if system_parts:
            kwargs["system"] = anthropic_system_param(
                system_parts, prompt_caching=self.prompt_caching
            )
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking

        if self.streaming:
            # Prefer streaming for long requests; accumulate events and normalize into
            # the same {text, tool_calls, usage} shape as non-streaming.
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

    def __repr__(self) -> str:
        return f"AnthropicModel(model='{self.model}')"


def _normalize_openai_tool_response(message: Any, usage: Any = None) -> Dict[str, Any]:
    text = message.content or ""
    tool_calls = []
    for call in getattr(message, "tool_calls", []) or []:
        raw_args = call.function.arguments or "{}"
        raw_args_str = raw_args if isinstance(raw_args, str) else str(raw_args)
        raw_args_err: str | None = None
        try:
            arguments = json.loads(raw_args_str)
        except Exception as exc:
            # Keep arguments empty so schema validation triggers a clear error,
            # but preserve raw args for agent self-correction.
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
    }


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

    # OpenAI automatic prompt caching: cached_tokens lives under prompt_tokens_details.
    pt_details = getattr(usage, "prompt_tokens_details", None)
    ct = getattr(pt_details, "cached_tokens", None) if pt_details is not None else None
    if ct is None:
        ct = getattr(usage, "cached_tokens", None)
    if ct is not None:
        normalized["cache_read_input_tokens"] = int(ct)

    if not normalized and hasattr(usage, "model_dump"):
        return usage.model_dump()
    if not normalized and hasattr(usage, "dict"):
        return usage.dict()
    return normalized or None


def _merge_stream_usage(start_usage: Any, delta_output_tokens: int) -> Optional[Dict[str, Any]]:
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


def anthropic_split_messages(
    chat_template: List[Dict[str, Any]],
) -> tuple[List[str], List[Dict[str, Any]]]:
    system_parts: List[str] = []
    messages: List[Dict[str, Any]] = []
    for msg in chat_template:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        else:
            messages.append(msg)
    return system_parts, messages


def anthropic_system_param(
    system_parts: List[str], prompt_caching: bool
) -> Union[str, List[Dict[str, Any]]]:
    if prompt_caching:
        # Cache only the first (static) system block; later blocks are dynamic.
        blocks = [
            {
                "type": "text",
                "text": system_parts[0],
                "cache_control": {"type": "ephemeral"},
            }
        ]
        blocks += [{"type": "text", "text": p} for p in system_parts[1:]]
        return blocks
    return "\n\n".join(system_parts)


def anthropic_apply_message_cache_breakpoint(
    messages: List[Dict[str, Any]], prompt_caching: bool
) -> List[Dict[str, Any]]:
    if not prompt_caching or len(messages) < 2:
        return messages

    updated = list(messages)

    # Add a cache breakpoint on the penultimate user message so the growing
    # tool-call / tool-result history is served from cache on each iteration.
    n_user = 0
    for i in range(len(updated) - 1, -1, -1):
        if updated[i]["role"] == "user":
            n_user += 1
            if n_user == 2:
                m = updated[i]
                c = m.get("content", "")
                if isinstance(c, str):
                    # Convert to block format so cache_control can be attached.
                    updated[i] = dict(
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
                        new_c = c[:-1] + [
                            {**last, "cache_control": {"type": "ephemeral"}}
                        ]
                        updated[i] = dict(m, content=new_c)
                break

    return updated


def _openai_tool_to_anthropic_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    fn = tool["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn["parameters"],
    }


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
            model[len("openai-"):], profile="openai", reasoning=reasoning
        )

    if model.lower().startswith("anthropic-"):
        return AnthropicModel(model[len("anthropic-"):], reasoning=reasoning)

    if model.lower().startswith("oaic-"):
        return OpenAICompModel(
            model[len("oaic-"):], profile="oaic", reasoning=reasoning
        )

    raise ValueError(
        f"Unknown model '{model}'. Use a recognized prefix: "
        "openai-, anthropic-, or oaic-."
    )
