# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Bounded author → review loop"""

from ._check import CheckResult, run_check, workspace_tree, workspace_diff
from ._state import (
    LoopPaths,
    LoopRule,
    LoopScore,
    LoopSummary,
    ReviewItem,
    get_loop_paths,
    init_state,
    write_state,
    parse_rules,
)
from ._guard import ProtectedFileGuard, GuardedBashTool
from ._prompts import (
    AUTHOR_SYSTEM,
    REVIEW_SYSTEM,
    build_author_task,
    build_review_task,
    extract_pending_items,
    extract_status,
)
from ._runner import PromptLoopRunner, prompt_loop
