# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Resolve a recipe's pipeline context (model class, display name, manifest)
from its on-disk folder. The single unit of identity throughout the
export/evaluate/install stack is ``source_dir: Path`` — the recipe folder
that contains ``manifest.yaml``. ``model_id`` is a display-only string
derived from ``source_dir.name`` and is only accepted at the CLI top
layer, where :func:`resolve_recipe_dir` converts it into a folder path.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_MODELS_ROOT


def resolve_recipe_dir(target: str | Path) -> Path:
    """Convert a CLI target (folder path or installed model id) to a folder path.

    Called once at the CLI top layer; everything downstream operates on
    ``Path``. Accepts:

    * A **folder path** (str or Path, absolute or relative). The folder
      must contain a ``manifest.yaml``.
    * A **bare installed model id** (e.g. ``"mobilenet_v2"``) — resolved
      to ``<installed_pkg>/models/<id>/``.
    * A **bare folder name** that matches a directory in the current
      working directory — resolved as a folder path. Installed model ids
      always win over cwd folders of the same name.

    Display names (``"MobileNetV2"``) are NOT accepted.
    """
    if isinstance(target, Path) or looks_like_path(str(target)):
        source_dir = Path(target).resolve()
        if not source_dir.is_dir():
            raise ValueError(
                f"{target!r} is not a directory. Point at a recipe folder "
                "that contains manifest.yaml."
            )
    else:
        target_str = str(target)
        if target_str in MODEL_IDS:
            source_dir = QAIHM_MODELS_ROOT / target_str
        elif (cwd_folder := Path(target_str)).is_dir():
            source_dir = cwd_folder.resolve()
        else:
            raise ValueError(
                f"{target_str!r} is not an installed model id and no folder "
                f"of that name exists in the current directory. Either use a "
                "known model id (see `qai-hub-models models` for the list) "
                "or pass a folder path (e.g. ./my_model/)."
            )
    if not (source_dir / "manifest.yaml").exists():
        raise ValueError(
            f"{source_dir} does not contain a manifest.yaml — cannot resolve "
            "as a model recipe."
        )
    return source_dir


def looks_like_path(target: str) -> bool:
    """Return True if *target* is meant as a filesystem path, not a bare id."""
    return (
        "/" in target
        or "\\" in target
        or target.startswith((".", "~"))
        or Path(target).is_absolute()
    )


def import_recipe_module(source_dir: Path) -> Any:
    """Import the recipe package at *source_dir* and return its module.

    For recipes inside the installed package, this returns the already-loaded
    ``qai_hub_models.models.<id>`` module. For standalone folders, the folder's
    parent is added to ``sys.path`` so ``import <folder_name>`` works — and
    any sub-imports the recipe uses (e.g. ``<folder_name>.external_repos.<repo>``)
    resolve consistently.
    """
    try:
        rel = source_dir.resolve().relative_to(QAIHM_MODELS_ROOT.resolve())
        if len(rel.parts) == 1:
            return importlib.import_module(f"qai_hub_models.models.{rel.parts[0]}")
    except ValueError:
        pass
    parent = str(source_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return importlib.import_module(source_dir.name)


def resolve_manifest(source_dir: Path) -> QAIHMModelManifest:
    """Load the manifest for the recipe at *source_dir*."""
    return QAIHMModelManifest.from_yaml(source_dir / "manifest.yaml")


def resolve_model_cls(source_dir: Path) -> Any:
    """Return the ``Model`` class exported from the recipe's ``__init__.py``."""
    return import_recipe_module(source_dir).Model


def resolve_model_app_cls(source_dir: Path) -> Any | None:
    """Return the ``App`` class exported from the recipe's package, or ``None``."""
    return getattr(import_recipe_module(source_dir), "App", None)


def resolve_model_display_name(source_dir: Path) -> str:
    """Resolve the human-readable model name from ``manifest.yaml``."""
    return resolve_manifest(source_dir).name or source_dir.name


def resolve_model_id(source_dir: Path) -> str:
    """Return the recipe's model id — its folder name."""
    return source_dir.name
