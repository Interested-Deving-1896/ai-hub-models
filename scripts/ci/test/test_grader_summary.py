# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for the markdown the CI grader summary renders."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_CI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CI_DIR))
from grader_summary import DEFAULT_METRIC, render  # noqa: E402


def _write_grade(directory: Path, name: str, **overrides: object) -> None:
    grade = {
        "input_file": f"/some/dir/{name}_eval.json",
        "metric": "Grace2",
        "grader_model": "Qwen/Qwen3.6-35B-A3B",
        "num_items": 100,
        "score_pct": 64.5,
        "total_points": 645,
        "max_points": 1000,
        "num_unparsed": 0,
        "reference_score_pct": 71.0,
        "summary_items": [
            "Blatant factual errors (most responses)",
            "Duplicated words mid-sentence (several)",
        ],
        "category_scores": {
            "coding": {"score_pct": 45.0, "points": 45, "num_scored": 10},
            "knowledge": {"score_pct": 90.0, "points": 90, "num_scored": 10},
        },
    }
    grade.update(overrides)
    (directory / f"{name}_eval_grade.json").write_text(json.dumps(grade))


def test_render_is_empty_without_grade_files(tmp_path: Path) -> None:
    assert render(str(tmp_path)) == []


def test_render_main_table(tmp_path: Path) -> None:
    _write_grade(tmp_path, "qwen3_0_6b_8_elite_w4a16")
    lines = render(str(tmp_path))

    assert lines[0] == "## Grace2 Response Grading"
    assert "| Model | Grace2 (on device) | Grace2 (ref) | Summary |" in lines
    # Device score, reference score, and a bullet list of the summary, one row.
    row = next(line for line in lines if line.startswith("| qwen3_0_6b"))
    assert "64.5%" in row
    assert "71.0%" in row
    assert "<ul><li>Blatant factual errors (most responses)</li>" in row
    assert "Grader: `Qwen/Qwen3.6-35B-A3B`" in lines


def test_render_leaves_reference_cell_empty_when_absent(tmp_path: Path) -> None:
    """No manifest baseline (or collector never ran) means a blank cell, not 0%."""
    _write_grade(tmp_path, "no_ref", reference_score_pct=None)
    row = next(line for line in render(str(tmp_path)) if line.startswith("| no_ref"))
    assert row == "| no_ref | 64.5% |  | " + row.split("| ")[-1]
    assert "0.0%" not in row


def test_render_category_table_is_one_collapsed_table(tmp_path: Path) -> None:
    """Every run shares one hidden table, with categories as columns."""
    _write_grade(tmp_path, "model_a")
    _write_grade(
        tmp_path,
        "model_b",
        category_scores={
            "coding": {"score_pct": 55.0, "points": 55, "num_scored": 10},
            "reasoning": {"score_pct": 20.0, "points": 20, "num_scored": 10},
        },
    )
    lines = render(str(tmp_path))
    text = "\n".join(lines)

    assert lines.count("<details>") == 1
    assert "<summary><b>Score by category</b> (weakest first)</summary>" in lines
    # Rows in both tables start with the model name, so scope to the breakdown.
    breakdown = lines[lines.index("<details>") :]
    # Categories are columns, ordered weakest mean first: reasoning (20) then
    # coding (50) then knowledge (90).
    header = next(line for line in breakdown if line.startswith("| Model | n |"))
    assert header == "| Model | n | reasoning | coding | knowledge |"
    # One row per run, and a run missing a category gets a placeholder, not 0%.
    row_a = next(line for line in breakdown if line.startswith("| model_a |"))
    assert row_a == "| model_a | 100 | — | 45.0% | 90.0% |"
    row_b = next(line for line in breakdown if line.startswith("| model_b |"))
    assert row_b == "| model_b | 100 | 20.0% | 55.0% | — |"
    # The breakdown comes after the main table.
    assert text.index("| Model | Grace2 (on device)") < text.index("<details>")


def test_render_falls_back_when_metric_absent(tmp_path: Path) -> None:
    _write_grade(tmp_path, "old_run", metric=None)
    lines = render(str(tmp_path))
    assert lines[0] == f"## {DEFAULT_METRIC} Response Grading"


def test_render_handles_missing_summary_and_categories(tmp_path: Path) -> None:
    _write_grade(tmp_path, "no_extras", summary_items=[], category_scores={})
    lines = render(str(tmp_path))
    assert any(line.endswith("| — |") for line in lines)
    assert "<details>" not in lines
    assert not any(line.startswith("| Model | n |") for line in lines)


def test_render_surfaces_grader_failures(tmp_path: Path) -> None:
    """An unrated response is the grader's fault and must not be buried."""
    _write_grade(tmp_path, "broken", num_unparsed=3)
    lines = render(str(tmp_path))
    failure = next(line for line in lines if "GRADER FAILURE" in line)
    assert "`broken`" in failure
    assert "3 response(s)" in failure
    assert "scored 0" in failure
    # Visible, not hidden behind the collapsed category table.
    assert lines.index(failure) < lines.index("<details>")


def test_render_dedupes_grader_model(tmp_path: Path) -> None:
    _write_grade(tmp_path, "run_a")
    _write_grade(tmp_path, "run_b")
    lines = render(str(tmp_path))
    assert lines[-1] == "Grader: `Qwen/Qwen3.6-35B-A3B`"


def test_render_escapes_pipes_in_summary_items(tmp_path: Path) -> None:
    _write_grade(tmp_path, "pipes", summary_items=["Markdown | leakage (one)"])
    row = next(line for line in render(str(tmp_path)) if line.startswith("| pipes"))
    assert "Markdown \\| leakage (one)" in row
