#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Render a markdown report summarizing LLM grader output JSON files.

Reads ``*_grade.json`` files (produced by
``qai_hub_models.scripts.llm.grade_responses --output-json``) from the given
directory and prints a markdown table to stdout. Intended for use in a
GitHub Actions step that appends to ``$GITHUB_STEP_SUMMARY``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

# Fallback label for a grade file written before the metric was recorded in it.
DEFAULT_METRIC = "Grace"


def _cell(text: str) -> str:
    """Escape a string for use inside a markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def _summary_cell(items: list[str]) -> str:
    """Render summary items as a bullet list inside one table cell.

    GitHub renders inline HTML in table cells, so a real ``<ul>`` works where a
    markdown list would not.
    """
    if not items:
        return "—"
    bullets = "".join(f"<li>{_cell(item)}</li>" for item in items)
    return f"<ul>{bullets}</ul>"


def _run_name(grade: dict[str, Any], path: str) -> str:
    source = grade.get("input_file") or path
    return os.path.basename(source).replace("_eval.json", "").replace(".json", "")


def _score_cell(value: Any) -> str:
    """A percentage, or an empty cell when the score is absent."""
    if value is None:
        return ""
    return f"{float(value):.1f}%"


def _category_order(grades: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Every category seen, weakest mean score first.

    The table is wide, so the categories most likely to need attention go
    leftmost rather than being scrolled to.
    """
    totals: dict[str, list[float]] = {}
    for _, grade in grades:
        for name, entry in (grade.get("category_scores") or {}).items():
            totals.setdefault(name, []).append(float(entry.get("score_pct", 0.0)))
    return sorted(totals, key=lambda name: sum(totals[name]) / len(totals[name]))


def _category_table(grades: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """One row per run, one column per category. Empty if no category resolved."""
    categories = _category_order(grades)
    if not categories:
        return []
    header = " | ".join(_cell(name) for name in categories)
    lines = [
        f"| Model | n | {header} |",
        "|-------|--:|" + "------:|" * len(categories),
    ]
    for path, grade in grades:
        scores = grade.get("category_scores") or {}
        cells = [
            _score_cell((scores.get(name) or {}).get("score_pct")) or "—"
            for name in categories
        ]
        lines.append(
            f"| {_cell(_run_name(grade, path))} | {grade.get('num_items', 0)} | "
            + " | ".join(cells)
            + " |"
        )
    return lines


def _failure_lines(grades: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """A visible line per run the grader failed to rate every response of."""
    lines = []
    for path, grade in grades:
        if not grade.get("num_unparsed"):
            continue
        lines.append(
            f"- **GRADER FAILURE** in `{_run_name(grade, path)}`: "
            f"{grade['num_unparsed']} response(s) produced no readable rating and "
            f"were scored 0. That score is a floor, not a measurement."
        )
    return lines


def render(directory: str) -> list[str]:
    grades = []
    for path in sorted(glob.glob(os.path.join(directory, "*_grade.json"))):
        with open(path) as f:
            grades.append((path, json.load(f)))
    if not grades:
        return []

    metric = next(
        (grade.get("metric") for _, grade in grades if grade.get("metric")),
        DEFAULT_METRIC,
    )
    lines = [
        f"## {metric} Response Grading",
        "",
        f"| Model | {metric} (on device) | {metric} (ref) | Summary |",
        "|-------|------:|------:|---------|",
    ]
    for path, grade in grades:
        lines.append(
            f"| {_cell(_run_name(grade, path))} "
            f"| {_score_cell(grade.get('score_pct'))} "
            f"| {_score_cell(grade.get('reference_score_pct'))} "
            f"| {_summary_cell(grade.get('summary_items') or [])} |"
        )

    failures = _failure_lines(grades)
    if failures:
        lines.extend(["", *failures])

    category_table = _category_table(grades)
    if category_table:
        lines.extend(
            [
                "",
                "<details>",
                "<summary><b>Score by category</b> (weakest first)</summary>",
                "",
                *category_table,
                "</details>",
            ]
        )

    graders = sorted({str(grade.get("grader_model", "unknown")) for _, grade in grades})
    lines.extend(["", "Grader: " + ", ".join(f"`{name}`" for name in graders)])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory containing *_grade.json files.")
    args = parser.parse_args()
    for line in render(args.directory):
        print(line)


if __name__ == "__main__":
    main()
