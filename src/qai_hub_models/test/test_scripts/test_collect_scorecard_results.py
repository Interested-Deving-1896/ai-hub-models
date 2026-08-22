# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for the toolchain-diff baseline helper in collect_scorecard_results.

Focused on the specific fallback path that keeps the "Previous" toolchain
column meaningful when S3 is unavailable or has not seen a run for this
deployment yet.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from qai_hub_models.scorecard.results.yaml import (
    ComponentNamesYaml,
    GraphNamesYaml,
    ToolVersionsByPathYaml,
)
from qai_hub_models.scripts import collect_scorecard_results as mod


def test_previous_tool_versions_uses_s3_baseline_when_available(
    tmp_path: Path,
) -> None:
    """When S3 has a previous same-deployment run, its tool-versions wins."""
    s3_yaml = tmp_path / "s3-tool-versions.yaml"
    s3_yaml.write_text("tool_versions:\n  qnn_context_binary:\n    qairt: 2.99.0.abc\n")
    manifest = mock.MagicMock(run_id="prev-run-id", date="2026-07-02")

    with (
        mock.patch.object(mod, "find_latest_run", return_value=manifest),
        mock.patch.object(mod, "download_single_artifact", return_value=s3_yaml),
    ):
        result = mod._load_previous_tool_versions("dev")

    versions = list(result.tool_versions.values())
    assert versions and versions[0].qairt is not None
    assert "2.99.0" in versions[0].qairt.full_version_with_flavor


@pytest.mark.parametrize(
    ("find_latest", "download_result"),
    [
        # No prior same-deployment run in S3 (new deployment, sandbox with no creds).
        (mock.DEFAULT, None),
        # Manifest exists but artifact missing (uploaded before we added it).
        (mock.MagicMock(run_id="old-run-id", date="2026-06-01"), None),
        # S3 raises (auth failure, network drop) — must not propagate.
        (RuntimeError("no creds"), None),
    ],
    ids=["no-manifest", "manifest-but-no-artifact", "s3-exception"],
)
def test_previous_tool_versions_falls_back_to_intermediates(
    find_latest: object, download_result: Path | None
) -> None:
    """Every miss path must return a valid ToolVersionsByPathYaml, not crash."""
    if isinstance(find_latest, Exception):
        patch = mock.patch.object(mod, "find_latest_run", side_effect=find_latest)
    elif find_latest is mock.DEFAULT:
        patch = mock.patch.object(mod, "find_latest_run", return_value=None)
    else:
        patch = mock.patch.object(mod, "find_latest_run", return_value=find_latest)
    with (
        patch,
        mock.patch.object(
            mod, "download_single_artifact", return_value=download_result
        ),
    ):
        result = mod._load_previous_tool_versions("dev")
    assert isinstance(result, ToolVersionsByPathYaml)


def test_names_survive_for_in_scope_model_that_produced_no_artifacts() -> None:
    """A model this run listed but produced no names for keeps its committed entry.

    An unconditional per-model clear lost these permanently; see 5bb159b896.
    """
    committed_components = ComponentNamesYaml({"measured": ["A"], "flaked": ["B"]})
    committed_graphs = GraphNamesYaml({"measured_A": ["g_old"], "flaked_B": ["g_keep"]})
    fresh_components = ComponentNamesYaml({"measured": ["A"]})
    fresh_graphs = GraphNamesYaml({"measured_A": ["g_new"]})

    mod.drop_names_with_replacements(
        committed_components,
        committed_graphs,
        fresh_components,
        fresh_graphs,
        ["measured", "flaked"],
    )

    assert committed_components.get("flaked") == ["B"]
    assert committed_graphs.get("flaked", "B") == ["g_keep"]
    # Cleared, ready for the merge that follows.
    assert committed_components.get("measured") is None
    assert committed_graphs.get("measured", "A") is None


def test_dropped_component_leaves_no_stale_graph_key() -> None:
    """A component removed from a recipe must not leave its graph-name key behind."""
    committed_components = ComponentNamesYaml({"m": ["A", "B"]})
    committed_graphs = GraphNamesYaml({"m_A": ["g_a"], "m_B": ["g_b"]})
    fresh_components = ComponentNamesYaml({"m": ["A"]})
    fresh_graphs = GraphNamesYaml({"m_A": ["g_a"]})

    mod.drop_names_with_replacements(
        committed_components,
        committed_graphs,
        fresh_components,
        fresh_graphs,
        ["m"],
    )

    assert committed_graphs.get("m", "B") is None
