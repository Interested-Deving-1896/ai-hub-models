# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""The built-in Grace prompt set: 10 categories x 10 prompts.

Grace is "Grading Response Accuracy Evaluation".

``grace<version>.jsonl`` holds one record per line::

    {"idx": 0, "category": "knowledge", "prompt": "What is gravity?"}
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import zip_longest

# Grace ("Grading Response Accuracy Evaluation") is the graded free-form response
# metric: this prompt set plus the grader rubric in grader.py. The version bumps
# whenever either changes enough that scores stop being comparable.
GRACE_VERSION = 2
# Task/dataset name, and the label the score is reported under.
GRACE_TASK_NAME = f"grace{GRACE_VERSION}"
GRACE_METRIC_NAME = f"Grace{GRACE_VERSION}"
# Convenience alias on the command line: always the latest version.
GRACE_TASK_ALIAS = "grace"
# Public documentation for the prompt set, the grader rubric, and how the score
# is computed. Published alongside the score as the metric's dataset link.
GRACE_DOC_URL = (
    "https://github.com/qualcomm/ai-hub-models/blob/main/tutorials/llm/grace.md"
)

GRACE_PROMPTS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{GRACE_TASK_NAME}.jsonl")
)

# The image + question set, graded the same way.
MULTIMODAL_TASK_NAME = "multimodal_prompts"
# Tasks whose evaluator generates free-form responses and grades them, as
# opposed to the forward-only metrics (wikitext, mmlu, ...).
PROMPT_TASKS: frozenset[str] = frozenset({GRACE_TASK_NAME, MULTIMODAL_TASK_NAME})


@dataclass(frozen=True)
class EvalPrompt:
    idx: int
    category: str
    prompt: str


def load_eval_prompts(path: str | os.PathLike | None = None) -> list[EvalPrompt]:
    """Load a prompt set from ``.jsonl`` records."""
    path = GRACE_PROMPTS_PATH if path is None else path
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    try:
        prompts = [
            EvalPrompt(int(r["idx"]), str(r["category"]), str(r["prompt"]))
            for r in records
        ]
    except KeyError as e:
        raise ValueError(
            f"{path} is missing the {e.args[0]!r} field: every record needs "
            "idx, category and prompt."
        ) from e

    if [p.idx for p in prompts] != list(range(len(prompts))):
        raise ValueError(
            f"{path} must hold contiguous idx values 0..{len(prompts) - 1} in "
            "order: idx is the join key for device responses and grader summaries."
        )
    return prompts


def resolve_task_name(task: str) -> str:
    """Map the version-less ``grace`` alias onto the current Grace task name."""
    return GRACE_TASK_NAME if task == GRACE_TASK_ALIAS else task


def load_default_eval_prompts() -> list[str]:
    """The built-in accuracy set as plain prompt strings, in ``idx`` order."""
    return [p.prompt for p in load_eval_prompts()]


def default_categories_by_idx() -> dict[int, str]:
    """Map ``idx`` to category for the built-in set."""
    return {p.idx: p.category for p in load_eval_prompts()}


def default_categories_by_prompt() -> dict[str, str]:
    """Map prompt text to category for the built-in set.

    Device runs record only ``{idx, prompt, output}``, so the category has to be
    recovered when grading. Text is the safer key of the two: ``idx`` on device
    is a position in whatever prompt list was staged.
    """
    return {p.prompt: p.category for p in load_eval_prompts()}


def select_balanced(prompts: list[EvalPrompt], count: int) -> list[EvalPrompt]:
    """Take ``count`` prompts spread evenly across categories, in ``idx`` order.

    Records are grouped by category, so a prefix slice would silently drop whole
    categories from a shortened run.
    """
    if count >= len(prompts):
        return prompts
    by_category: dict[str, list[EvalPrompt]] = defaultdict(list)
    for prompt in prompts:
        by_category[prompt.category].append(prompt)
    round_robin = [
        prompt
        for row in zip_longest(*by_category.values())
        for prompt in row
        if prompt is not None
    ]
    return sorted(round_robin[:count], key=lambda p: p.idx)
