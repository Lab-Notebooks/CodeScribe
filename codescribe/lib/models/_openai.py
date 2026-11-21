# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

from __future__ import annotations

import importlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from codescribe.lib.models._base import Model
from codescribe.lib.models._usage import attr, normalize_openai_usage


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is not set")
    return value


def _summary_texts(reasoning: Any) -> List[str]:
    """Text from a `reasoning` field: a string, or an object with a summary."""
    if reasoning is None:
        return []
    if isinstance(reasoning, str):
        return [reasoning]
    summary = getattr(reasoning, "summary", None)
    if isinstance(summary, str):
        return [summary]
    if isinstance(summary, list):
        return [
            s if isinstance(s, str) else getattr(s, "text", None) or "" for s in summary
        ]
    return []


def _content_texts(content: Any) -> Tuple[List[str], List[str]]:
    """(text, reasoning) pieces from a `content` field (string or block list)."""
    if isinstance(content, str):
        return [content], []
    texts: List[str] = []
    reasoning: List[str] = []
    for block in content if isinstance(content, list) else []:
        btype = attr(block, "type")
        btext = attr(block, "text") or attr(block, "summary")
        if btype == "text" and btext:
            texts.append(btext)
        elif btype in ("reasoning", "summary_text") and btext:
            reasoning.append(btext)
    return texts, reasoning


def _join_unique(parts: List[str]) -> str:
    seen: List[str] = []
    for p in (p.strip() for p in parts if p):
        if p and p not in seen:
            seen.append(p)
    return "\n\n".join(seen)


def _tool_call(call_id: Any, name: Any, raw: Any) -> Dict[str, Any]:
    """A normalized tool call; unparseable arguments are kept for self-correction."""
    raw = raw or "{}"
    raw = raw if isinstance(raw, str) else str(raw)
    call: Dict[str, Any] = {"id": call_id, "name": name}
    try:
        call["arguments"] = json.loads(raw)
    except Exception as exc:
        call["arguments"] = {}
        call["_raw_arguments"] = raw
        call["_raw_arguments_error"] = f"{type(exc).__name__}: {exc}"
    return call


def _reply(
    text: str, tool_calls: List[Dict[str, Any]], usage: Any, reasoning: str
) -> Dict[str, Any]:
    return {
        "text": text,
        "tool_calls": tool_calls,
        "usage": usage,
        "reasoning": reasoning,
        "reasoning_blocks": [{"type": "reasoning", "text": reasoning}]
        if reasoning
        else [],
    }


class OpenAICompModel(Model):
    """OpenAI Chat Completions, or any compatible endpoint (profile "oaic")."""

    api = "openai"
    outputs = 1

    def __init__(
        self, model: str, profile: str = "oaic", reasoning: bool = False
    ) -> None:
        super().__init__(model, reasoning)
        openai = importlib.import_module("openai")
        self.profile = profile
        self.reasoning_effort: Optional[str] = (
            "high" if self.reasoning_enabled else None
        )

        if profile == "openai":
            self.apikey = _required_env("OPENAI_API_KEY")
            self.baseurl = None
            self.provider = None
        elif profile == "oaic":
            self.baseurl = _required_env("OPENAI_COMP_BASEURL")
            self.provider = _required_env("OPENAI_COMP_PROVIDER")
            self.apikey = _required_env("OPENAI_COMP_APIKEY")
        else:
            raise ValueError(
                f"Unknown OpenAI profile '{profile}'. Use 'openai' or 'oaic'."
            )
        self.pipeline = openai.OpenAI(api_key=self.apikey, base_url=self.baseurl)

    def chat(self, chat_template: List[Dict[str, Any]]) -> str:
        reply = self.complete(self.request_kwargs(chat_template))
        return "\n\n".join(p for p in (reply["reasoning"], reply["text"]) if p)

    def chat_with_tools(
        self, chat_template: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return self.complete(self.request_kwargs(chat_template, tools))

    def format_tool_result_messages(
        self,
        tool_calls: List[Dict[str, Any]],
        outputs: List[str],
        reasoning_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        reasoning = "\n\n".join(
            b.get("text", "") for b in reasoning_blocks or [] if b.get("text")
        )
        assistant = {
            "role": "assistant",
            "content": reasoning or None,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    },
                }
                for call in tool_calls
            ],
        }
        results = [
            {"role": "tool", "tool_call_id": call["id"], "content": output}
            for call, output in zip(tool_calls, outputs)
        ]
        return [assistant, *results]

    def request_kwargs(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                m
                for m in messages
                if m.get("role") in {"system", "user", "assistant", "tool"}
            ],
            "max_tokens": self.max_tokens,
            "n": self.outputs,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def complete(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Stream when possible, else a plain create() call."""
        reply = self.stream(kwargs)
        if reply is None:
            response = self.pipeline.chat.completions.create(**kwargs)
            reply = self.normalize_message(
                response.choices[0].message, getattr(response, "usage", None)
            )
        self.last_usage = reply["usage"]
        return reply

    def stream(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # include_usage adds a final usage chunk; strict compatible servers
        # reject it, so retry without before giving up on streaming.
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

        # Reasoning arrives as token-sized deltas: concatenate per source.
        text: List[str] = []
        block_reasoning: List[str] = []
        delta_reasoning: List[str] = []
        calls: Dict[int, Dict[str, Any]] = {}
        usage = None
        try:
            for chunk in stream:
                usage = getattr(chunk, "usage", None) or usage
                choices = getattr(chunk, "choices", None) or []
                delta = getattr(choices[0], "delta", None) if choices else None
                if delta is None:
                    continue

                texts, reasoning = _content_texts(getattr(delta, "content", None))
                text += texts
                block_reasoning += reasoning
                delta_reasoning += _summary_texts(getattr(delta, "reasoning", None))

                for call in getattr(delta, "tool_calls", None) or []:
                    index = getattr(call, "index", None)
                    entry = calls.setdefault(
                        len(calls) if index is None else index,
                        {"id": None, "name": None, "arguments": ""},
                    )
                    if getattr(call, "id", None):
                        entry["id"] = call.id
                    function = getattr(call, "function", None)
                    if getattr(function, "name", None):
                        entry["name"] = function.name
                    entry["arguments"] += getattr(function, "arguments", None) or ""
        except Exception:
            return None

        return _reply(
            "".join(text),
            [
                _tool_call(c["id"], c["name"], c["arguments"])
                for _, c in sorted(calls.items())
            ],
            normalize_openai_usage(usage),
            _join_unique(["".join(delta_reasoning), "".join(block_reasoning)]),
        )

    def normalize_message(self, message: Any, usage: Any = None) -> Dict[str, Any]:
        texts, block_reasoning = _content_texts(getattr(message, "content", None))
        reasoning = (
            _summary_texts(getattr(message, "reasoning", None)) + block_reasoning
        )
        tool_calls = [
            _tool_call(call.id, call.function.name, call.function.arguments)
            for call in getattr(message, "tool_calls", None) or []
        ]
        return _reply(
            "".join(texts),
            tool_calls,
            normalize_openai_usage(usage),
            _join_unique(reasoning),
        )

    def __repr__(self) -> str:
        return (
            f"OpenAICompModel(model='{self.model}', profile='{self.profile}', "
            f"outputs={self.outputs}, max_tokens={self.max_tokens})"
        )
