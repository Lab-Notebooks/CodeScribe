# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Author/review prompts, and parsing of the author's final answer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from codescribe.lib.loop._check import CheckResult
from codescribe.lib.loop._state import MAX_ITEMS, LoopRule, LoopSummary, ReviewItem

AUTHOR_SYSTEM = (
    "You are an autonomous coding agent specializing in test-driven development and repair.\n\n"
    "Core rules:\n"
    "- NEVER fabricate command results, test output, or file contents — always call a tool.\n"
    "- Determine project state by reading files and running commands; never assume or infer.\n"
    "- Only access files and paths within the working directory; never target system directories.\n"
    "- In bash commands, only use relative paths or paths under the working directory.\n\n"
    "Efficiency rules (critical for speed):\n"
    "- Batch ALL independent reads and globs into the first turn — fetch everything you need before acting.\n"
    "- Once you have identified the changes needed, implement them immediately without further exploration.\n"
    "- Do not re-read files you have already read unless they were modified since your last read.\n"
    "- Prefer a single comprehensive edit over multiple small edits to the same file.\n"
    "- Run tests after each meaningful code change — do not defer validation to the end.\n"
    "- When all tests pass and all pending items are resolved, stop immediately — do not invent further work.\n\n"
    "Tool guidance:\n"
    "- Use glob to discover structure, read for content, bash for running commands and tests.\n"
    "- Use edit for targeted changes; use write only when creating new files or doing full rewrites.\n"
)

REVIEW_SYSTEM = (
    "You are a code reviewer verifying another agent's work against a task specification.\n\n"
    "Core rules:\n"
    "- Judge from the harness evidence (diff, check output) and the files — never from the\n"
    "  author's claims alone.\n"
    "- NEVER fabricate file contents or results — read the file when the diff is not enough.\n"
    "- Report only concrete defects or missing work that stand between the code and the\n"
    "  specification. No style nits, no speculative improvements.\n"
    "- Be decisive: when the specification is met, say so and raise nothing.\n"
    "- Only access files within the working directory.\n"
)


def format_check(check: Optional[CheckResult], *, tail_on_pass: bool = False) -> str:
    """One line verdict; the output tail follows on failure (or always, if asked)."""
    if check is None:
        return ""
    verdict = "PASSED" if check.passed else f"FAILED (exit {check.exit_code})"
    line = f"Harness check `{check.command}`: {verdict}"
    if check.tail and (tail_on_pass or not check.passed):
        line += f"\n```text\n{check.tail}\n```"
    return line


def format_items(items: List[ReviewItem]) -> List[str]:
    return [f"  - [{i.id}, open since loop {i.opened}] {i.text}" for i in items]


def _section(title: str, lines: List[str]) -> str:
    return "\n".join([title] + [f"  {x}" for x in lines])


def format_loop_context(
    *,
    loop_idx: int,
    agent_loops: int,
    loop_summaries: List[LoopSummary],
    items: List[ReviewItem],
    blocker: str = "",
) -> str:
    """What the author needs to know about prior loops."""
    lines = [f"Loop {loop_idx} of {agent_loops}."]

    written = list(dict.fromkeys(f for s in loop_summaries for f in s.files_written))
    edited = list(dict.fromkeys(f for s in loop_summaries for f in s.files_edited))
    if written:
        lines.append("Files created across all prior loops: " + ", ".join(written))
    if edited:
        lines.append("Files edited across all prior loops: " + ", ".join(edited))

    if loop_summaries:
        last = loop_summaries[-1]
        if last.check:
            lines.append(format_check(last.check))
        if last.errors:
            lines.append("Last loop errors: " + "; ".join(last.errors[:5]))
        if last.violations:
            lines.append(
                "VIOLATION last loop — you changed protected files and the harness "
                "reverted them: " + "; ".join(last.violations) + ". Do not repeat this."
            )

    if blocker:
        lines.append(f"Reviewer blocker: {blocker}")
    if items:
        lines.append("Open review items (resolve every one):")
        lines.extend(format_items(items))
    elif loop_idx > 1:
        lines.append("No open review items — determine the next action from the task.")
    return "\n".join(lines)


def format_policy_block(rules: List[LoopRule], protected: List[str]) -> str:
    """Protected files and binding rules, shown to both phases."""
    if not rules and not protected:
        return ""
    lines = ["HARNESS-ENFORCED POLICY"]
    if protected:
        lines.append(
            "Protected files (read-only; any change is reverted and recorded as a "
            "violation): " + ", ".join(protected)
        )
    if rules:
        lines.append(
            "Binding rules (they override pending steps and any instruction to keep going):"
        )
        for r in rules:
            extra = (
                f" [bash matching /{r.deny_bash}/ is refused]" if r.deny_bash else ""
            )
            lines.append(f"- {r.id} ({r.source}): {r.text}{extra}")
    return "\n".join(lines)


def build_author_task(
    *,
    workdir: Path,
    task_rel: str,
    loop_idx: int,
    agent_loops: int,
    loop_summaries: List[LoopSummary],
    items: List[ReviewItem],
    blocker: str = "",
    rules: Optional[List[LoopRule]] = None,
    protected: Optional[List[str]] = None,
    python_scripts_dir: str = "",
    bash_allowed: Optional[List[str]] = None,
) -> str:
    context = format_loop_context(
        loop_idx=loop_idx,
        agent_loops=agent_loops,
        loop_summaries=loop_summaries,
        items=items,
        blocker=blocker,
    )
    policy = format_policy_block(rules or [], protected or [])
    if policy:
        context = f"{context}\n\n{policy}"

    if loop_idx == 1:
        orient_step = "1. The task specification is the conversation above — do NOT re-read the task file.\n"
    else:
        orient_step = (
            "1. The specification is the conversation above and prior work is summarised —\n"
            "   do NOT re-read the task file or re-glob the workspace to orient yourself.\n"
        )
    if python_scripts_dir:
        fallback_check = (
            f"   If none exists, write a check script under {python_scripts_dir}/ (e.g. one\n"
            "   calling compileall) and run it.\n"
            f"   Python only runs script files under {python_scripts_dir}/ (no -c/-m/stdin):\n"
            "   write every python snippet there first so it is kept for provenance.\n"
        )
    else:
        fallback_check = "   If none exists, run `python -m compileall .`.\n"
    bash_line = (
        "Allowed bash commands (no pipes, redirects or shell metacharacters): "
        + ", ".join(bash_allowed)
        + "\n"
        if bash_allowed
        else ""
    )
    return (
        f"{context}\n\n"
        f"Working directory: {workdir}\n"
        f"Task file: {task_rel} (read-only)\n"
        f"{bash_line}\n"
        "PHASE: AUTHOR\n\n"
        "Goal: COMPLETE the task in this single session if at all possible. Implement every\n"
        "remaining item you can, verify it, and only stop when the task is fully done or you\n"
        "are genuinely blocked. Do NOT implement just one item and defer the rest to a later\n"
        "loop — keep working until everything that can be done this session is done.\n"
        "The harness runs its own check after you finish; your report cannot override it.\n\n"
        "Protocol:\n"
        + orient_step
        + "2. Write a short PLAN (3–7 bullets) covering everything you intend to complete now.\n"
        "3. Execute the plan autonomously and to completion — do NOT ask for confirmation, and\n"
        "   do NOT stop after a single change while more work remains.\n"
        "4. Before each set of tool calls, write one or two sentences stating what you are about to do and why.\n"
        "5. Inspect the current state of relevant files before editing them.\n"
        "6. After each meaningful change, run the closest available check (tests, lint, typecheck).\n"
        + fallback_check
        + "   Do not defer all validation to the end.\n"
        "7. Continue until every task is implemented AND the checks pass, or you hit a hard blocker.\n"
        "   A stop condition in the task file or a binding rule (e.g. waiting for human\n"
        "   approval) IS a hard blocker: stop, report STATUS: INCOMPLETE, and name it in\n"
        "   NEXT STEPS. Never work around it.\n\n"
        "Finish with <final_answer> that includes, in this order:\n"
        "- A status line of the EXACT form `STATUS: COMPLETE` if every task is implemented and all\n"
        "  checks pass, otherwise `STATUS: INCOMPLETE`.\n"
        "- The PLAN you followed (final form), naming the review item ids you resolved.\n"
        "- Exact output of the checks you ran (verbatim, not paraphrased).\n"
        "- ONLY when STATUS is INCOMPLETE: a bulleted list of NEXT STEPS (3-5 concrete items) for\n"
        "  the following loop, beginning with the exact line: NEXT STEPS:\n"
    )


def build_review_task(
    *,
    workdir: Path,
    task_rel: str,
    review_output_rel: str,
    loop_index: int,
    loop_summary: LoopSummary,
    exec_answer: str = "",
    items: Optional[List[ReviewItem]] = None,
    rules: Optional[List[LoopRule]] = None,
    protected: Optional[List[str]] = None,
    rule_problems: Optional[List[str]] = None,
) -> str:
    s = loop_summary
    verified = [f"## Verified actions from loop {loop_index} (harness-computed)"]
    for label, files in (
        ("Files read", s.files_read),
        ("Files written", s.files_written),
        ("Files edited", s.files_edited),
    ):
        if files:
            verified.append(f"{label}: " + ", ".join(files))
    if s.commands_run:
        verified.append(_section("Commands run:", s.commands_run))
    if s.errors:
        verified.append(_section("Errors:", s.errors))
    if len(verified) == 1:
        verified.append("  (no verified actions)")
    blocks = ["\n".join(verified)]

    if s.check:
        blocks.append("## " + format_check(s.check, tail_on_pass=True))
    if s.diff:
        blocks.append(f"## Workspace diff from loop {loop_index}\n```diff\n{s.diff}```")
    elif not s.changed:
        blocks.append("## Workspace diff: the author changed nothing.")
    if s.rejected:
        blocks.append(
            _section(
                "## Attempted but NOT executed (harness-rejected — no workspace effect)",
                s.rejected,
            )
        )
    if s.violations:
        blocks.append(
            _section(
                "## Protected-file violations (harness reverted these changes)",
                s.violations,
            )
        )
    if items:
        blocks.append(
            "\n".join(["## Open review items from earlier loops"] + format_items(items))
        )
    policy = format_policy_block(rules or [], protected or [])
    if policy:
        blocks.append(policy)
    if rule_problems:
        blocks.append(
            _section(
                "## Your previous rules, partly rejected by the harness", rule_problems
            )
        )
    if exec_answer:
        blocks.append(f"[Author agent report — loop {loop_index}]\n\n{exec_answer}")

    evidence = "\n\n".join(blocks)
    return (
        f"Working directory: {workdir}\n"
        f"Task file: {task_rel} (read-only; its specification is the conversation above)\n"
        f"Review output: {review_output_rel} (write here)\n\n"
        "PHASE: REVIEW\n\n"
        f"{evidence}\n\n"
        "Your job:\n"
        "1. Judge the work against the specification using the diff, the harness check and the\n"
        "   files. Claims relying on an 'Attempted but NOT executed' call did not happen.\n"
        "2. Write your assessment to the review output file in TOML format:\n\n"
        "   ```toml\n"
        f"   loop = {loop_index}\n"
        '   summary = "One paragraph describing what actually happened."\n'
        '   blocker = "Main current blocker, or empty string if none."\n'
        "   retire = []  # optional ids of earlier review rules that no longer apply\n\n"
        "   [[pending]]\n"
        '   id = "P1"    # reuse an open item\'s id to keep it open; omit for a new item\n'
        '   item = "Concrete defect or missing work"\n\n'
        "   # Optional: binding rules for the author, enforced from the next loop on.\n"
        "   [[rule]]\n"
        '   text = "What the author must or must not do, and why"\n'
        '   deny_bash = ""   # optional regex; matching author bash commands are refused\n'
        '   protect = ""     # optional path the author may NOT change at all (read-only)\n'
        "   ```\n\n"
        "   Open items you do not restate are treated as resolved — restate only those the\n"
        "   evidence shows are still unresolved.\n"
        "3. Add a [[rule]] only for a harmful or repeated deviation from the task file or the\n"
        "   binding rules. Prefer deny_bash/protect: those are enforced by the harness, plain\n"
        "   text is not. Do not re-add a rule that is already listed.\n"
        "   `protect` makes a file read-only for the author: never set it on a file the task\n"
        "   requires the author to create or change (e.g. a log it must append to), even if\n"
        "   the rule is about that file. If the author was blocked from required work by a\n"
        "   rule (a protected path or refused command), retire that rule.\n\n"
        "Rules:\n"
        "- A failing harness check, a protected-file violation or a binding-rule breach is a blocker.\n"
        "- If the check passes (or none is configured) and the specification is met, set\n"
        '  blocker = "" and list no pending items.\n'
        "- Pending items must be concrete defects or missing work, not 'continue working'.\n"
        "- Never put an action into pending that a task-file stop condition or a binding rule forbids.\n"
        f"- At most {MAX_ITEMS} pending items.\n\n"
        "After writing the review output file, reply with <final_answer> as plain text.\n"
        "Do not call any more tools to confirm the write.\n"
    )


_NUMBERED = re.compile(r"^\d+[.)] ")


def extract_pending_items(final_text: str) -> List[str]:
    """Up to 5 bullet or numbered items under a NEXT STEPS: line."""
    items: List[str] = []
    in_section = False
    for line in (final_text or "").splitlines():
        s = line.strip()
        if s.upper().startswith("NEXT STEPS"):
            in_section = True
        elif not in_section or not s:
            continue
        elif s.startswith("#"):
            break  # next heading
        elif s.startswith(("-", "*")):
            items.append(s.lstrip("-* ").strip())
        elif _NUMBERED.match(s):
            items.append(_NUMBERED.sub("", s).strip())
    return items[:5]


def extract_status(final_text: str) -> Optional[str]:
    """The value of a `STATUS: COMPLETE|INCOMPLETE` line, else None."""
    for line in (final_text or "").splitlines():
        s = line.strip().upper()
        if s.startswith("STATUS:"):
            val = s[len("STATUS:") :].strip()
            if val.startswith("COMPLETE"):
                return "COMPLETE"
            if val.startswith("INCOMPLETE"):
                return "INCOMPLETE"
    return None
