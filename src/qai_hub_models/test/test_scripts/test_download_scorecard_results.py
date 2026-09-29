# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for scorecard-history S3 filtering."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from qai_hub_models.scorecard.history import ScorecardManifest
from qai_hub_models.scripts import download_scorecard_results as mod


def _mk(run_id: str, deployment: str, run_name: str, date: str) -> ScorecardManifest:
    return ScorecardManifest(
        run_id=run_id, date=date, deployment=deployment, run_name=run_name
    )


def test_find_latest_run_returns_prior_weekly_run_of_same_deployment() -> None:
    """The issue's Previous cell must be last week's weekly-dev scorecard, even
    when a week of nightly/manual uploads (and a manual dev dispatch that was
    mislabelled weekly-prod, tetracode #21470) sit newer in S3. Before this
    fix those pushed the weekly run out of the 25-manifest download window.
    """
    manifests = {
        f"scorecard-history/{9000 + i}-adhoc/manifest.json": _mk(
            str(9000 + i), "prod", "adhoc", "2026-09-16"
        )
        for i in range(30)
    }
    manifests["scorecard-history/8500-weekly-prod/manifest.json"] = _mk(
        "8500", "dev", "weekly-prod", "2026-09-15"
    )
    manifests["scorecard-history/8000-weekly-dev/manifest.json"] = _mk(
        "8000", "dev", "weekly-dev", "2026-09-10"
    )

    def fake_download(bucket: str, key: str, dest: str, verbose: bool) -> None:
        manifests[key].to_json(dest)

    listing = [mock.Mock(key=k) for k in manifests]
    with (
        mock.patch.object(mod, "get_qaihm_s3_or_exit", return_value=("bkt", None)),
        mock.patch.object(
            mod, "list_s3_files_in_folder_recursive", return_value=listing
        ),
        mock.patch.object(mod, "s3_download", side_effect=fake_download) as download,
    ):
        result = mod.find_latest_run("dev")

    assert result is not None
    assert result.run_id == "8000"
    download.assert_called_once()


def test_find_latest_run_with_scheduled_ids_ignores_mislabelled_nightlies() -> None:
    """Nightly/manual prod runs were stored as weekly-prod too, so the key can't
    tell them apart. With GitHub's scheduled IDs, Prod Previous must be last
    week's scheduled run (tetracode #21470: 09-19 vs 09-12, not a nightly).
    """
    manifests = {
        "scorecard-history/9500-weekly-prod/manifest.json": _mk(
            "9500", "prod", "weekly-prod", "2026-09-21"
        ),
        "scorecard-history/9000-weekly-prod/manifest.json": _mk(
            "9000", "prod", "weekly-prod", "2026-09-19"
        ),
        "scorecard-history/8000-weekly-prod/manifest.json": _mk(
            "8000", "prod", "weekly-prod", "2026-09-12"
        ),
    }

    def fake_download(bucket: str, key: str, dest: str, verbose: bool) -> None:
        manifests[key].to_json(dest)

    with (
        mock.patch.object(mod, "get_qaihm_s3_or_exit", return_value=("bkt", None)),
        mock.patch.object(
            mod, "s3_file_exists", side_effect=lambda _b, key: key in manifests
        ),
        mock.patch.object(mod, "s3_download", side_effect=fake_download),
    ):
        result = mod.find_latest_run(
            "prod",
            exclude_run_id="9000",
            # 7777 is a scheduled dev run: no weekly-prod manifest, so skipped.
            scheduled_run_ids=["9000", "7777", "8000"],
        )

    assert result is not None
    assert result.run_id == "8000"


def test_download_single_artifact_key_matches_upload_layout(tmp_path: Path) -> None:
    """The S3 key must be {PREFIX}/{run_id}-{run_name}/{artifact}, matching
    upload_scorecard_history.upload_scorecard_to_s3. A key mismatch here
    silently returns None on every lookup, and the toolchain diff / Scorecard
    Context grid fall back to intermediates without any error.
    """
    manifest = _mk("29003645353", "dev", "weekly-dev", "2026-07-09")
    dest = tmp_path / "tool-versions.yaml"

    with (
        mock.patch.object(mod, "get_qaihm_s3_or_exit", return_value=("bkt", None)),
        mock.patch.object(mod, "s3_file_exists", return_value=True) as exists,
        mock.patch.object(mod, "s3_download") as download,
    ):
        result = mod.download_single_artifact(manifest, "tool-versions.yaml", dest)

    expected_key = "scorecard-history/29003645353-weekly-dev/tool-versions.yaml"
    exists.assert_called_once_with("bkt", expected_key)
    download.assert_called_once_with("bkt", expected_key, dest, verbose=False)
    assert result == dest
