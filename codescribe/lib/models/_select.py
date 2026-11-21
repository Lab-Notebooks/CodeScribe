# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

from __future__ import annotations

from codescribe.lib.models._anthropic import AnthropicModel
from codescribe.lib.models._base import Model
from codescribe.lib.models._openai import OpenAICompModel


def set_neural_model(model: str, reasoning: bool = False) -> Model:
    """Backend for a prefixed model string: anthropic-*, openai-* or oaic-*."""
    prefix, _, name = model.partition("-")
    prefix = prefix.lower()
    if prefix == "anthropic":
        return AnthropicModel(name, reasoning=reasoning)
    if prefix in ("openai", "oaic"):
        return OpenAICompModel(name, profile=prefix, reasoning=reasoning)
    raise ValueError(
        f"Unknown model '{model}'. Use a recognized prefix: "
        "openai-, anthropic-, or oaic-."
    )
