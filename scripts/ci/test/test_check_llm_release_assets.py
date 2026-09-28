# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for reporting partial asset-upload shard coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import check_llm_release_assets as mod

MATRIX = json.dumps(
    [
        {"split_name": "llm-1-of-3", "models": "a"},
        {"split_name": "llm-2-of-3", "models": "b"},
        {"split_name": "llm-3-of-3", "models": "c"},
    ]
)


def _artifact_dirs(root: Path, split_names: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in split_names:
        (root / f"Scorecard LLM Perf Release Assets (123, {name})").mkdir()
    return root


def test_missing_shard_is_detected(tmp_path: Path) -> None:
    downloaded = _artifact_dirs(tmp_path / "dl", ["llm-1-of-3", "llm-3-of-3"])
    assert mod.missing_splits(mod.parse_expected(MATRIX), downloaded) == ["llm-2-of-3"]


def test_all_shards_present(tmp_path: Path) -> None:
    downloaded = _artifact_dirs(
        tmp_path / "dl", ["llm-1-of-3", "llm-2-of-3", "llm-3-of-3"]
    )
    assert mod.missing_splits(mod.parse_expected(MATRIX), downloaded) == []


def test_no_download_dir_means_every_shard_missing(tmp_path: Path) -> None:
    expected = mod.parse_expected(MATRIX)
    assert mod.missing_splits(expected, tmp_path / "absent") == expected


def test_unknown_matrix_reports_nothing_missing(tmp_path: Path) -> None:
    assert mod.parse_expected("") == []
    assert mod.missing_splits([], tmp_path) == []


def test_load_models_reads_model_keys(tmp_path: Path) -> None:
    assets = tmp_path / "release-assets.yaml"
    assets.write_text("models:\n  qwen3_0_6b:\n    precisions: {}\n  llama_v3: {}\n")
    assert mod.load_models(assets) == ["llama_v3", "qwen3_0_6b"]


@pytest.mark.parametrize("contents", ["models: {}\n", ""])
def test_load_models_empty_is_no_models(tmp_path: Path, contents: str) -> None:
    assets = tmp_path / "release-assets.yaml"
    assets.write_text(contents)
    assert mod.load_models(assets) == []


def test_load_models_missing_file(tmp_path: Path) -> None:
    assert mod.load_models(tmp_path / "nope.yaml") == []


def test_main_writes_outputs_and_never_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing shard must not fail the job -- it only reports."""
    downloaded = _artifact_dirs(tmp_path / "dl", ["llm-1-of-3"])
    assets = tmp_path / "release-assets.yaml"
    assets.write_text("models:\n  qwen3_0_6b:\n    precisions: {}\n")
    output = tmp_path / "gh_output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_llm_release_assets",
            "--release-assets",
            str(assets),
            "--downloaded-dir",
            str(downloaded),
            "--split-matrix",
            MATRIX,
            "--github-output",
            str(output),
        ],
    )

    assert mod.main() == 0
    written = output.read_text()
    assert "fresh_asset_models=qwen3_0_6b\n" in written
    assert "missing_splits=llm-2-of-3,llm-3-of-3\n" in written
