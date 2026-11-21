# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Optional

from codescribe.lib.models._base import Model
from codescribe.lib.models._usage import merge_stream_usage, normalize_anthropic_usage

_CACHE_5M = {"type": "ephemeral"}
_CACHE_1H = {"type": "ephemeral", "ttl": "1h"}


def _with_cache_breakpoint(message: Dict[str, Any]) -> Dict[str, Any]:
    """`message` with cache_control on its last content block."""
    content = message.get("content", "")
    if isinstance(content, str):
        return dict(
            message,
            content=[{"type": "text", "text": content, "cache_control": _CACHE_5M}],
        )
    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict) and "cache_control" not in last:
            return dict(
                message, content=content[:-1] + [{**last, "cache_control": _CACHE_5M}]
            )
    return message


def _normalize_response(response: Any, usage: Any = None) -> Dict[str, Any]:
    texts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    reasoning: List[str] = []
    # Thinking blocks keep Anthropic's wire format: they are echoed back verbatim.
    reasoning_blocks: List[Dict[str, Any]] = []
    for block in response.content:
        if block.type == "thinking":
            reasoning.append(block.thinking or "")
            rb: Dict[str, Any] = {"type": "thinking", "thinking": block.thinking or ""}
            if getattr(block, "signature", None):
                rb["signature"] = block.signature
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
        "reasoning": "\n\n".join(t for t in reasoning if t).strip(),
        "reasoning_blocks": reasoning_blocks,
    }


class AnthropicModel(Model):
    """Anthropic Messages API, with prompt caching and optional adaptive thinking."""

    api = "anthropic"

    def __init__(self, model: str, reasoning: bool = False) -> None:
        super().__init__(model, reasoning)
        anthropic = importlib.import_module("anthropic")
        self.apikey = os.getenv("ANTHROPIC_API_KEY")
        if not self.apikey:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.base_url = os.getenv("ANTHROPIC_BASE_URL")
        self.thinking = (
            {"type": "adaptive", "display": "summarized"}
            if self.reasoning_enabled
            else None
        )

        client_kwargs: Dict[str, Any] = {
            "api_key": self.apikey,
            # The 1h TTL is set per block; this header only helps older gateways.
            "default_headers": {"anthropic-beta": "extended-cache-ttl-2025-04-11"},
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = anthropic.Anthropic(**client_kwargs)

    def chat(self, chat_template: List[Dict[str, Any]]) -> str:
        kwargs = self.request_kwargs(chat_template)
        streamed = self.stream(kwargs)
        if streamed is not None:
            return streamed["text"]
        response = self.client.messages.create(**kwargs)
        self.last_usage = normalize_anthropic_usage(getattr(response, "usage", None))
        return next((b.text for b in response.content if b.type == "text"), "")

    def chat_with_tools(
        self, chat_template: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        kwargs = self.request_kwargs(chat_template, tools)
        streamed = self.stream(kwargs)
        if streamed is None:
            response = self.client.messages.create(**kwargs)
            self.last_usage = normalize_anthropic_usage(
                getattr(response, "usage", None)
            )
            return _normalize_response(response, self.last_usage)

        final = streamed["final_message"]
        if final is None:
            return {
                "text": streamed["text"],
                "tool_calls": [],
                "usage": self.last_usage,
            }
        reply = _normalize_response(final, self.last_usage)
        reply["text"] = reply["text"] or streamed["text"]
        return reply

    def stream(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Stream a request (required by the SDK for long requests).

        Returns {text, final_message} and sets last_usage, or None if streaming
        is unsupported.
        """
        try:
            stream = self.client.messages.stream(**kwargs)
        except Exception:
            return None

        text: List[str] = []
        start_usage = None
        output_tokens = 0
        final_message = None
        with stream as s:
            for event in s:
                et = getattr(event, "type", None)
                if et == "message_start":
                    start_usage = getattr(
                        getattr(event, "message", None), "usage", None
                    )
                elif et == "message_delta":
                    du = getattr(event, "usage", None)
                    if du is not None:
                        output_tokens = int(getattr(du, "output_tokens", 0) or 0)
                elif et == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) == "text_delta":
                        text.append(getattr(delta, "text", "") or "")
            try:
                final_message = s.get_final_message()
            except Exception:
                pass

        self.last_usage = merge_stream_usage(start_usage, output_tokens)
        return {"text": "".join(text), "final_message": final_message}

    def format_tool_result_messages(
        self,
        tool_calls: List[Dict[str, Any]],
        outputs: List[str],
        reasoning_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        # Thinking blocks must precede tool_use blocks, verbatim.
        assistant = list(reasoning_blocks or []) + [
            {
                "type": "tool_use",
                "id": c["id"],
                "name": c["name"],
                "input": c["arguments"],
            }
            for c in tool_calls
        ]
        results = [
            {"type": "tool_result", "tool_use_id": c["id"], "content": output}
            for c, output in zip(tool_calls, outputs)
        ]
        return [
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": results},
        ]

    def request_kwargs(
        self,
        chat_template: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        system = [m["content"] for m in chat_template if m["role"] == "system"]
        messages = [m for m in chat_template if m["role"] != "system"]

        # Rolling 5m breakpoint on the second-to-last user turn: already frozen,
        # so turn N reads what turn N-1 wrote. Tools and system are cached at 1h
        # (longer TTLs must precede shorter ones in the prefix).
        user_turns = [i for i, m in enumerate(messages) if m["role"] == "user"]
        if len(messages) >= 2 and len(user_turns) >= 2:
            i = user_turns[-2]
            messages[i] = _with_cache_breakpoint(messages[i])

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = [
                {"type": "text", "text": system[0], "cache_control": _CACHE_1H},
                *[{"type": "text", "text": p} for p in system[1:]],
            ]
        if tools is not None:
            kwargs["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]
            if kwargs["tools"]:
                kwargs["tools"][-1]["cache_control"] = _CACHE_1H
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        return kwargs
