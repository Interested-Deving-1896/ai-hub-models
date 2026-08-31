# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""``qai-hub-models generate-files`` implementation.

Regenerates the auto-generated files that live inside a recipe folder
from its ``manifest.yaml``:

    external_repos/__init__.py  (only when the manifest declares external_repos)
    README.md

Both files are safe to blow away and regenerate — they contain no
hand-written content. The command is the external-user-facing subset
of the internal ``run_codegen.py`` script (which additionally generates
``export.py``, ``evaluate.py``, ``conftest.py``, and
``test_generated.py`` — none of which ship with a standalone recipe).

The target is a folder path (``my_model``) or an installed model id
(``yolov8_det``). Both must contain a ``manifest.yaml`` at the root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.asset_loaders import UNPUBLISHED_MODEL_WARNING
from qai_hub_models.utils.export.context import resolve_recipe_dir
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT

_TEMPLATES_DIR = QAIHM_PACKAGE_ROOT / "scripts" / "templates"
_HEADER = (
    "# ---------------------------------------------------------------------\n"
    "# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.\n"
    "# SPDX-License-Identifier: BSD-3-Clause\n"
    "# ---------------------------------------------------------------------\n"
    "# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY."
)


def _write_external_repos_init(folder: Path, manifest: QAIHMModelManifest) -> Path:
    """Render the external_repos/__init__.py bootstrap for *folder*.

    Does nothing (returns the path anyway) when the manifest has no
    ``external_repos:`` declaration.
    """
    external_repos_dir = folder / "external_repos"
    init_path = external_repos_dir / "__init__.py"

    if not manifest.external_repos:
        return init_path

    external_repos_dir.mkdir(exist_ok=True)
    env = Environment(
        loader=FileSystemLoader(_TEMPLATES_DIR),
        keep_trailing_newline=True,
    )
    template = env.get_template("external_repos_init_template.j2")

    render_args: dict[str, str] = {"header": _HEADER}
    if folder.parent.name == "templates":
        render_args["template_name"] = folder.name
    else:
        render_args["model_id"] = folder.name

    init_path.write_text(template.render(**render_args))
    return init_path


def write_readme(
    folder: Path,
    manifest: QAIHMModelManifest,
    out_dir: Path | None = None,
    hf_front_matter: str | None = None,
    hf_repo_id: str | None = None,
) -> Path:
    """Render the model README from the manifest into ``<out_dir>/README.md``.

    Standalone / external recipes (``status: unset``) get an external-flavored
    README whose commands take the folder name (``qai-hub-models install
    <folder>``, ``qai-hub-models demo <folder>``, ``qai-hub-models export
    <folder>``) and which drops the catalog-only Quick Start block, the
    workbench.aihub.qualcomm.com model-page link, and the unpublished
    warning banner. In-tree recipes get the existing catalog-flavored
    README.

    Parameters
    ----------
    folder
        Recipe folder the manifest came from.
    manifest
        Parsed manifest driving every template variable.
    out_dir
        Where to write README.md. Defaults to *folder*; ``upload-to-hf``
        points it at a staging directory instead.
    hf_front_matter
        Rendered YAML metadata for a Hugging Face model card. Left None by
        ``generate-files`` so the on-disk README carries no front matter.
    hf_repo_id
        Hugging Face repo id. When set, the card gains a section showing how to
        register and run the published recipe.

    Returns
    -------
    Path
        The README path written.
    """
    # The scripts package (and its templates) is not available in release
    # builds, so import here to avoid failures: generate-files is dev-only.
    from qai_hub_models.scripts.generate_model_readme import (
        _has_machine_gated_entries,
        get_shared_template_args,
    )

    if manifest.headline is None:
        raise ValueError(
            f"{folder}/manifest.yaml is missing required field for README "
            "generation (headline). Fill it in and re-run generate-files."
        )

    is_external = manifest.status is MODEL_STATUS.UNSET

    env = Environment(
        loader=FileSystemLoader(_TEMPLATES_DIR),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("model_readme_template.j2")

    supported_precisions = None
    if manifest.supported_precisions and (
        manifest.can_use_quantize_job or manifest.separate_quantize_script
    ):
        supported_precisions = [str(p) for p in manifest.supported_precisions]

    template_vars = get_shared_template_args(manifest)
    template_vars.update(
        {
            "model_status": manifest.status.value,
            "unpublished_model_warning": UNPUBLISHED_MODEL_WARNING,
            "model_headline": manifest.headline.strip("."),
            "model_package": manifest.get_package_name(),
            "model_assets_shareable": not manifest.restrict_model_sharing,
            "default_runtime": manifest.default_runtime.value,
            "default_precision": str(manifest.default_precision),
            "supported_precisions": supported_precisions,
            "separate_quantize_script": manifest.separate_quantize_script,
            "has_machine_gated_entries": _has_machine_gated_entries(manifest),
            "python_version_gte": manifest.python_version_greater_than_or_equal_to,
            "python_version_lt": manifest.python_version_less_than,
            "include_example_and_usage": not manifest.skip_example_usage,
            "has_on_target_demo": manifest.has_on_target_demo,
            "readme_export_device": manifest.default_device
            if manifest.requires_aot_prepare
            else None,
            "local_device_deployment": manifest.local_device_deployment,
            "readme_install_system_deps": manifest.readme_install_system_deps,
            "is_external": is_external,
            "hf_front_matter": hf_front_matter,
            "hf_repo_id": hf_repo_id,
        }
    )

    readme_path = (out_dir or folder) / "README.md"
    readme_path.write_text(template.module.render(**template_vars))  # type: ignore[attr-defined]
    return readme_path


def generate_files(target: str) -> list[Path]:
    """Regenerate the auto-generated files for *target*.

    Parameters
    ----------
    target
        Folder path (``my_model``) or an installed model id
        (``yolov8_det``).

    Returns
    -------
    list[Path]
        The paths of every file (re)written.
    """
    folder = resolve_recipe_dir(target)
    manifest = QAIHMModelManifest.from_yaml(folder / "manifest.yaml")

    written: list[Path] = []
    init_path = _write_external_repos_init(folder, manifest)
    if manifest.external_repos:
        written.append(init_path)
    written.append(write_readme(folder, manifest))
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qai-hub-models generate-files",
        description=(
            "Regenerate the auto-generated files (external_repos/__init__.py "
            "and README.md) inside a recipe folder from its manifest.yaml. "
            "Run this after editing manifest.yaml so downstream files stay in sync."
        ),
    )
    parser.add_argument(
        "target",
        help=(
            "Recipe folder path (e.g. my_model) or an installed model id "
            "(e.g. yolov8_det). The folder must contain a manifest.yaml."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point called from the lean-CLI dispatcher."""
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    written = generate_files(args.target)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
