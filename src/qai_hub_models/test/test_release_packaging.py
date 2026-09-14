# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Guard the release wheel's package include/exclude lists.

A published recipe whose source is dropped from the wheel still ships its
metadata (setuptools keeps files of excluded sub-packages as data files of an
ancestor package). The resulting metadata-only folder imports as an implicit
namespace package, so failures surface far downstream as a bare
``AttributeError: module '...' has no attribute 'Model'``.
"""

from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

SETUP_PY = Path(__file__).parents[2] / "setup.py"
MODELS_ROOT = Path(__file__).parents[1] / "models"


def _run_release_setup() -> tuple[dict[str, Any], dict[str, Any]]:
    """Exec setup.py as a release build; return (module globals, setup kwargs).

    ``setup()`` is called unconditionally at module scope, so it must be
    patched out rather than avoided via ``run_name``. ``find_packages`` walks
    the cwd, so this has to run from setup.py's own directory.
    """
    captured: dict[str, Any] = {}
    # setuptools_scm is a build-system requirement, so pip installs it into an
    # isolated build env and it is absent from the runtime venv CI tests in.
    scm = types.ModuleType("setuptools_scm")
    scm.get_version = lambda **kwargs: "0.0.1"  # type: ignore[attr-defined]
    prev_cwd = os.getcwd()
    os.chdir(SETUP_PY.parent)
    try:
        with (
            patch.dict(os.environ, {"QAIHM_RELEASE_BUILD": "1"}),
            patch.dict(sys.modules, {"setuptools_scm": scm}),
            patch("setuptools.setup", lambda **kw: captured.update(kw)),
        ):
            module_globals = runpy.run_path(str(SETUP_PY), run_name="__main__")
    finally:
        os.chdir(prev_cwd)
    return module_globals, captured


def _published_recipes_with_source() -> list[str]:
    out = []
    for d in sorted(MODELS_ROOT.iterdir()):
        manifest = d / "manifest.yaml"
        if not d.is_dir() or not manifest.exists():
            continue
        if not (d / "__init__.py").exists():
            continue
        with open(manifest) as f:
            if (yaml.safe_load(f) or {}).get("status") == "published":
                out.append(d.name)
    return out


@pytest.fixture(scope="module")
def release_setup() -> tuple[dict[str, Any], dict[str, Any]]:
    return _run_release_setup()


@pytest.mark.parametrize("model_id", _published_recipes_with_source())
def test_published_recipe_source_is_in_release_wheel(
    model_id: str, release_setup: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Every published recipe with source must survive the release exclude lists.

    Regression guard: the exclude patterns are fnmatch globs where ``*`` spans
    dots, so a bare ``<name>*`` suffix silently dropped ``sam2``/``sam3`` via
    the unpublished ``sam`` recipe.
    """
    packages = set(release_setup[1]["packages"])
    assert f"qai_hub_models.models.{model_id}" in packages


def test_release_excludes_are_not_prefix_globs(
    release_setup: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """No exclude pattern may end in ``<name>*``, which over-matches siblings."""
    patterns = release_setup[0]["_get_excluded_packages"]()
    offenders = [p for p in patterns if p.endswith("*") and not p.endswith(".*")]
    assert not offenders, (
        f"These patterns match sibling packages by prefix: {offenders}. "
        "Use both `<pkg>` and `<pkg>.*` instead of `<pkg>*`."
    )
