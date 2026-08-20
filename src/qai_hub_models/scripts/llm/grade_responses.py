#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
r"""CLI for grading LLM responses.

Reads a JSON file containing a list of items in the form::

    [
      {"idx": 0, "category": "knowledge", "prompt": "What is gravity?",
       "output": "Gravity is ..."},
      ...
    ]

The ``prompt`` is passed through to the grader as-is, and ``output`` is the
generated response to grade. Grading is delegated to
:mod:`qai_hub_models.models._shared.llm.grader.grader`.

A closing pass sends the rationales back to the grader model and condenses them
into at most five recurring failure modes, printed last and stored under
``summary_items`` in ``--output-json``. Pass ``--no-summary`` to skip it.

Example usage::

    python -m qai_hub_models.scripts.llm.grade_responses responses.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qai_hub_models.models._shared.llm.grader.grace import (
    GRACE_METRIC_NAME,
    default_categories_by_idx,
    default_categories_by_prompt,
)
from qai_hub_models.models._shared.llm.grader.grader import (
    DEFAULT_PROMPT_TEMPLATE,
    MAX_POINTS,
    GradeResult,
    ResponseGrader,
    resolve_device,
)


def _resolve_categories(items: list[dict]) -> list[str | None]:
    """Category per item, backfilled for response files that record none.

    Device runs write only ``{idx, prompt, output}``, so the category is looked
    up in the built-in prompt set by prompt text, then by ``idx``. None only for
    a prompt that is not in the built-in set at all.
    """
    by_prompt = default_categories_by_prompt()
    by_idx = default_categories_by_idx()
    return [
        item.get("category")
        or by_prompt.get(str(item.get("prompt", "")).strip())
        or by_idx.get(item.get("idx", -1))
        for item in items
    ]


def _category_scores(
    categories: list[str | None], results: list[GradeResult]
) -> dict[str, tuple[float, int, int]]:
    """Per-category (score_pct, points, num_scored), in first-seen order.

    Mirrors the overall score: an item the grader failed to rate scores 0 and
    stays in its category's denominator.
    """
    points: dict[str, int] = {}
    scored: dict[str, int] = {}
    for category, result in zip(categories, results, strict=True):
        if category is None:
            continue
        points[category] = points.get(category, 0) + result.points
        scored[category] = scored.get(category, 0) + 1
    return {
        name: (
            100.0 * points[name] / (MAX_POINTS * scored[name]),
            points[name],
            scored[name],
        )
        for name in points
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade LLM responses from a JSON file on a 0-10 rubric.",
    )
    parser.add_argument(
        "responses_json",
        type=str,
        nargs="?",
        help="Path to a JSON file: list of {idx, prompt, output} objects.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to grading prompt template (must contain {response}). "
        "If omitted, the default template is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.6-35B-A3B",
        help="HuggingFace model id to use as grader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for the grader model (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit grading on CPU. Without this, an unusable CUDA install is a "
        "hard error rather than a ~13-minute-per-file CPU fallback.",
    )
    parser.add_argument(
        "--check-device-only",
        action="store_true",
        help="Resolve and print the grader device, then exit. Lets CI fail fast on "
        "a misconfigured venv without loading a model or reading responses.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Grader model dtype.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write a machine-readable summary to this path.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the closing summary pass over the rationales.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the per-item score and the grader's rationale.",
    )
    args = parser.parse_args()

    # Resolved up front so a misconfigured venv fails before a model download.
    try:
        device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    except RuntimeError as e:
        raise SystemExit(str(e)) from None
    if args.check_device_only:
        print(f"grader device: {device}")
        return
    if args.responses_json is None:
        parser.error("responses_json is required unless --check-device-only is set.")

    if args.prompt_file:
        prompt_template = Path(args.prompt_file).read_text()
    else:
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    items = json.loads(Path(args.responses_json).read_text())
    if not items:
        raise ValueError(f"No items found in {args.responses_json}")
    print(f"Loaded {len(items)} items from {args.responses_json}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    grader = ResponseGrader(
        model_id=args.model,
        device=device,
        dtype=dtype_map[args.dtype],
        prompt_template=prompt_template,
        allow_cpu=args.allow_cpu,
    )

    summary = grader.grade(items, summary=not args.no_summary)

    if args.verbose:
        for item, result in zip(items, summary.results, strict=True):
            print(
                f"  idx={item['idx']}: {result.points:2d} pts"
                + ("  [skipped: empty response]" if result.skipped else "")
                + ("  [rating forced after token limit]" if result.forced else "")
                + ("  [GRADER FAILURE: no rating]" if not result.parsed else "")
            )
            if result.rationale:
                print(f"      {result.rationale}")

    print()
    print("=" * 60)
    print(f"Grader: {args.model}")
    print(f"Responses graded: {len(items)}")
    print("=" * 60)
    print(
        f"Overall score: {summary.score_pct:.1f}%  "
        f"({summary.total_points}/{summary.max_points} pts)"
    )
    if summary.num_forced:
        print(
            f"Note: {summary.num_forced} item(s) ran out of tokens before rating; "
            f"the rating was recovered by a forced second pass."
        )
    if summary.num_unparsed:
        print(
            f"GRADER FAILURE: {summary.num_unparsed} item(s) produced no readable "
            f"rating and were scored 0. The score above is a floor, not a "
            f"measurement — fix the grader and re-run before trusting it."
        )
    print()
    flawed = sorted(
        (result.points, item["idx"], result.rationale)
        for item, result in zip(items, summary.results, strict=True)
        if result.points < MAX_POINTS
    )
    if flawed:
        print(f"Items scoring below {MAX_POINTS}, worst first:")
        for points, idx, rationale in flawed:
            print(f"  - idx={idx} ({points}/{MAX_POINTS}): {rationale}")
        print()

    categories = _resolve_categories(items)
    category_scores = _category_scores(categories, summary.results)
    if category_scores:
        print("By category:")
        for name, (pct, pts, num) in sorted(
            category_scores.items(), key=lambda kv: kv[1][0]
        ):
            print(f"  {name:15s} {pct:5.1f}%  ({pts}/{MAX_POINTS * num} pts, n={num})")
        print()

    if summary.summary_items:
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        for number, text in enumerate(summary.summary_items, start=1):
            print(f"  {number}. {text}")
        print()

    if args.output_json:
        out = {
            "input_file": str(args.responses_json),
            "metric": GRACE_METRIC_NAME,
            "grader_model": args.model,
            "num_items": len(items),
            "score_pct": summary.score_pct,
            "total_points": summary.total_points,
            "max_points": summary.max_points,
            "num_unparsed": summary.num_unparsed,
            "num_forced": summary.num_forced,
            "summary_items": summary.summary_items,
            "category_scores": {
                name: {"score_pct": pct, "points": pts, "num_scored": num}
                for name, (pct, pts, num) in category_scores.items()
            },
            "items": [
                {
                    "idx": item["idx"],
                    "category": category,
                    "points": result.points,
                    "skipped": result.skipped,
                    "parsed": result.parsed,
                    "forced": result.forced,
                    "rationale": result.rationale,
                }
                for item, category, result in zip(
                    items, categories, summary.results, strict=True
                )
            ],
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"Wrote grading summary to {args.output_json}")


if __name__ == "__main__":
    main()
