# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Normalize provider usage objects into one flat dict shape (see TokenUsage.from_raw)."""

from __future__ import annotations

from typing import Any, Dict, Optional


def attr(obj: Any, *names: str) -> Any:
    """First non-None attribute (or key, for dicts) among `names`, else None."""
    if obj is None:
        return None
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _fallback(usage: Any, normalized: Dict[str, Any]) -> Any:
    if normalized:
        return normalized
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    return None


def normalize_openai_usage(usage: Any) -> Any:
    if usage is None or isinstance(usage, dict):
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

    # o-series nest reasoning tokens under completion_tokens_details; some
    # compatible providers report them at the top level.
    rt = attr(attr(usage, "completion_tokens_details"), "reasoning_tokens")
    if rt is None:
        rt = attr(usage, "reasoning_tokens")
    if rt is not None:
        normalized["reasoning_tokens"] = int(rt)

    details = attr(usage, "prompt_tokens_details", "input_tokens_details")
    ct = attr(details, "cached_tokens")
    if ct is None:
        ct = attr(usage, "cached_tokens")
    if ct is not None:
        normalized["cache_read_input_tokens"] = int(ct)

    cw = attr(details, "cache_write_tokens", "cache_creation_tokens")
    if cw is None:
        cw = attr(usage, "cache_write_tokens", "cache_creation_input_tokens")
    if cw is not None:
        normalized["cache_creation_input_tokens"] = int(cw)

    # OpenAI's prompt_tokens include cached tokens; Anthropic's input_tokens do
    # not. Subtract so TokenUsage.total does not double-count cache hits.
    cached_total = int(ct or 0) + int(cw or 0)
    if cached_total:
        for key in ("prompt_tokens", "input_tokens"):
            if key in normalized:
                normalized[key] = max(0, int(normalized[key]) - cached_total)

    return _fallback(usage, normalized)


def anthropic_cache_split(usage: Any) -> Dict[str, int]:
    """5m/1h cache-write breakdown from Anthropic's usage.cache_creation."""
    creation = attr(usage, "cache_creation")
    split: Dict[str, int] = {}
    for src, dst in (
        ("ephemeral_5m_input_tokens", "cache_creation_5m_input_tokens"),
        ("ephemeral_1h_input_tokens", "cache_creation_1h_input_tokens"),
    ):
        val = attr(creation, src)
        if isinstance(val, int):
            split[dst] = val
    return split


def normalize_anthropic_usage(usage: Any) -> Any:
    if usage is None or isinstance(usage, dict):
        return usage

    normalized = {}
    for src, dst in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("cache_creation_input_tokens", "cache_creation_input_tokens"),
        ("cache_read_input_tokens", "cache_read_input_tokens"),
        ("thinking_tokens", "reasoning_tokens"),
    ):
        value = getattr(usage, src, None)
        if value is not None:
            normalized[dst] = value
    normalized.update(anthropic_cache_split(usage))
    return _fallback(usage, normalized)


def merge_stream_usage(
    start_usage: Any, output_tokens: int
) -> Optional[Dict[str, Any]]:
    """Usage from Anthropic stream events: message_start (input + cache) and
    message_delta (output). Avoids get_final_message(), which can fail with
    extended thinking."""
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
        result.update(anthropic_cache_split(start_usage))
    if output_tokens:
        result["output_tokens"] = output_tokens
    return result or None
