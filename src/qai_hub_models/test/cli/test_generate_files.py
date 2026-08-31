# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for ``qai-hub-models generate-files`` README rendering.

Covers the external/internal split: external recipes (``status: unset``)
must get a README whose commands take the recipe folder path, drops the
catalog-only Quick Start block, drops the ``workbench.aihub.qualcomm.com``
model-page link, and drops the unpublished-warning banner. Internal
recipes (``status: pending`` / ``published``) keep the existing
catalog-flavored README.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qai_hub_models.cli.generate_files import write_readme
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest

_MINIMAL_MANIFEST = """\
applicable_scenarios:
- Medical Imaging
dataset:
- imagenet-1k
description: A test classifier.
domain: Computer Vision
form_factors:
- Phone
has_on_target_demo: true
headline: A test classifier.
id: {model_id}
license: https://example.com/LICENSE
license_type: bsd-3-clause
name: {model_name}
research_paper: https://arxiv.org/abs/1512.03385
research_paper_title: Test Paper
source_repo: https://github.com/example/test-model
{status_line}\
supported_precisions:
{precisions}\
tags:
- bu-auto
templates:
- imagenet_classifier
use_case: Image Classification
"""


def _write_recipe(
    folder: Path,
    model_id: str = "my_model",
    status_line: str = "",
    precisions: str = "- float\n",
) -> QAIHMModelManifest:
    """Author a minimal recipe folder + manifest at *folder* and return the manifest."""
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.yaml"
    manifest_path.write_text(
        _MINIMAL_MANIFEST.format(
            model_id=model_id,
            model_name="MyModel",
            status_line=status_line,
            precisions=precisions,
        )
    )
    return QAIHMModelManifest.from_yaml(manifest_path)


class TestExternalReadme:
    """External recipes: ``status:`` omitted from manifest → UNSET → external README."""

    def test_writes_without_status(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme_path = write_readme(recipe, manifest)
        assert readme_path == recipe / "README.md"
        assert readme_path.exists()

    def test_uses_bare_folder_name_for_install(self, tmp_path: Path) -> None:
        """A bare name resolves as a cwd folder, so no ./ prefix is advertised."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models install my_model" in readme
        assert "./my_model/" not in readme

    def test_uses_bare_folder_name_for_export(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models export my_model" in readme
        assert "./my_model/" not in readme

    def test_demo_uses_the_cli_command(self, tmp_path: Path) -> None:
        """`qai-hub-models demo`, not a python -m module path."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models demo my_model" in readme
        assert "python -m" not in readme

    def test_demo_command_has_no_quantize_flag(self, tmp_path: Path) -> None:
        """--quantize is a redundant alias for --precision; it must not be advertised."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe, precisions="- float\n- w8a8\n")
        readme = write_readme(recipe, manifest).read_text()
        assert "--quantize" not in readme

    def test_no_catalog_quickstart_commands(self, tmp_path: Path) -> None:
        """``qai-hub-models info/perf/numerics/fetch`` need a catalog id — external skips."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models info " not in readme
        assert "qai-hub-models perf " not in readme
        assert "qai-hub-models numerics " not in readme
        assert "qai-hub-models fetch " not in readme

    def test_no_quick_start_section(self, tmp_path: Path) -> None:
        """Quick Start duplicates Setup + Run CLI Demo + Export for external — drop it."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "## Quick Start" not in readme

    def test_no_catalog_model_page_link(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        # The title link (aihub.qualcomm.com/models/<id>) is catalog-only.
        # The generic workbench.aihub.qualcomm.com landing link is still
        # fine to reference elsewhere.
        assert "aihub.qualcomm.com/models/" not in readme

    def test_no_unpublished_warning_banner(self, tmp_path: Path) -> None:
        """UNSET is external's normal state; do not render the pending/unpublished banner."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "[!WARNING]" not in readme

    def test_missing_headline_raises(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest_path = recipe / "manifest.yaml"
        recipe.mkdir()
        manifest_path.write_text(
            _MINIMAL_MANIFEST.format(
                model_id="my_model",
                model_name="MyModel",
                status_line="",
                precisions="- float\n",
            ).replace("headline: A test classifier.\n", "")
        )
        manifest = QAIHMModelManifest.from_yaml(manifest_path)
        with pytest.raises(ValueError, match="headline"):
            write_readme(recipe, manifest)


class TestInternalReadme:
    """Internal recipes: explicit status → catalog-flavored README (existing behavior)."""

    def test_uses_bare_model_id_for_export(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe, status_line="status: pending\n")
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models export my_model" in readme
        assert "qai-hub-models export ./my_model/" not in readme

    def test_demo_uses_the_cli_command(self, tmp_path: Path) -> None:
        """In-tree cards use the same CLI command, with the bare model id."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe, status_line="status: pending\n")
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models demo my_model" in readme
        assert "python -m qai_hub_models.models.my_model.demo" not in readme

    def test_includes_catalog_quickstart(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe, status_line="status: pending\n")
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models info " in readme
        assert "qai-hub-models perf " in readme
        assert "qai-hub-models numerics " in readme

    def test_includes_catalog_model_page_link(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe, status_line="status: pending\n")
        readme = write_readme(recipe, manifest).read_text()
        assert "aihub.qualcomm.com/models/my_model" in readme

    def test_pending_shows_warning_banner(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe, status_line="status: pending\n")
        readme = write_readme(recipe, manifest).read_text()
        assert "[!WARNING]" in readme

    def test_install_uses_cli_not_extras(self, tmp_path: Path) -> None:
        """Setup uses ``qai-hub-models install <id>``, not ``pip install "qai-hub-models[<id>]"``."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe, status_line="status: pending\n")
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models install my_model" in readme
        assert 'pip install "qai-hub-models[my-model]"' not in readme
        assert 'pip install "qai-hub-models[my_model]"' not in readme
