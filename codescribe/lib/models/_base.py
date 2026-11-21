# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes")


class Model:
    """Chat backend driven by the agent.

    Subclasses normalize every reply to
    {text, tool_calls, usage, reasoning, reasoning_blocks}.
    """

    api = ""  # wire format of content blocks: "openai" | "anthropic"

    def __init__(self, model: str, reasoning: bool = False) -> None:
        self.model = model
        self.max_tokens = int(os.getenv("CODESCRIBE_MAX_TOKENS", "32768"))
        self.reasoning_enabled = reasoning or env_flag("CODESCRIBE_MODEL_REASONING")
        self.last_usage: Optional[Dict[str, Any]] = None

    def chat(self, chat_template: List[Dict[str, Any]]) -> str:
        raise NotImplementedError

    def chat_with_tools(
        self, chat_template: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def format_tool_result_messages(
        self,
        tool_calls: List[Dict[str, Any]],
        outputs: List[str],
        reasoning_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """The assistant tool-call turn plus its results, in this backend's format."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model='{self.model}')"
