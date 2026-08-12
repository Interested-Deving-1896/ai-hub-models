# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Heavy-side dispatcher for ``qai-hub-models <script> <target>`` subcommands.

Resolves *target* (a recipe folder path or a bare installed model id) to a
:class:`Path` once at the top layer, then hands it to the export/evaluate
pipelines and to the CLI parsers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from qai_hub_models.cli.generate_files import main as generate_files_main
from qai_hub_models.cli.install import main as install_main
from qai_hub_models.cli.validate import main as validate_main
from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.utils.args import evaluate_parser, export_parser
from qai_hub_models.utils.asset_loaders import check_unpublished_model_warning
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate.dispatch import select_evaluate_pipeline
from qai_hub_models.utils.export.context import (
    resolve_manifest,
    resolve_model_cls,
    resolve_recipe_dir,
)
from qai_hub_models.utils.export.dispatch import select_pipeline


def _confirm_run_ok(source_dir: Path) -> bool:
    """Return False iff the user declines the unpublished-recipe warning.

    Only ``status: published`` skips the prompt. Every other status
    (``unpublished``, ``pending``, or the ``unset`` default that external /
    newly-onboarded recipes ship with) triggers the confirmation prompt —
    running unreviewed recipe code should be an explicit user decision.
    """
    manifest = resolve_manifest(source_dir)
    if manifest.status is MODEL_STATUS.PUBLISHED:
        return True
    return check_unpublished_model_warning()


def build_export_parser_for(source_dir: Path) -> argparse.ArgumentParser:
    """Build the export parser for the recipe at *source_dir*."""
    manifest = resolve_manifest(source_dir)
    return export_parser(
        model_cls=resolve_model_cls(source_dir),
        export_fn=select_pipeline(source_dir),
        supported_precision_runtimes=manifest.get_supported_paths_for_export(),
        default_export_device=manifest.default_device,
        omit_precision=manifest.separate_quantize_script,
    )


def build_evaluate_parser_for(source_dir: Path) -> argparse.ArgumentParser:
    """Build the evaluate parser for the recipe at *source_dir*."""
    manifest = resolve_manifest(source_dir)
    model_cls = cast(type[BaseModel], resolve_model_cls(source_dir))
    supports_quant_cpu = (
        manifest.can_use_quantize_job and manifest.supports_quantization
    )
    return evaluate_parser(
        model_cls=model_cls,
        supported_dataset_classes=model_cls.get_eval_dataset_classes(),
        supported_precision_runtimes=manifest.get_supported_paths_for_export(),
        uses_quantize_job=supports_quant_cpu,
        num_calibration_samples=manifest.num_calibration_samples
        if manifest.num_calibration_samples
        else None,
        default_device=manifest.default_device,
    )


def run_model_script(model_id: str | Path, script: str, forwarded: list[str]) -> None:
    """Run the given script for the given recipe target.

    Parameters
    ----------
    model_id
        A recipe target — either a folder path (``./my_model/``, ``Path``,
        or path string) or a bare installed model id (``"yolov8_det"``).
        Kept named ``model_id`` for call-site compatibility. The lean CLI
        rejects ``script="install"``/``"generate-files"``/``"validate"``
        with a folder before this is called.
    script
        Script name: ``"export"``, ``"evaluate"``, ``"install"``,
        ``"generate-files"``, or ``"validate"``.
    forwarded
        Argv tail handed to the target's parser.
    """
    if script == "install":
        install_main([str(model_id), *forwarded])
        return

    if script == "generate-files":
        generate_files_main([str(model_id), *forwarded])
        return

    if script == "validate":
        validate_main([str(model_id), *forwarded])
        return

    source_dir = resolve_recipe_dir(model_id)
    if script == "export":
        parser = build_export_parser_for(source_dir)
        parser.prog = f"qai_hub_models export {source_dir.name}"
        args = parser.parse_args(forwarded)
        if not _confirm_run_ok(source_dir):
            return
        select_pipeline(source_dir)(**vars(args))
        return

    if script == "evaluate":
        parser = build_evaluate_parser_for(source_dir)
        parser.prog = f"qai_hub_models evaluate {source_dir.name}"
        args = parser.parse_args(forwarded)
        if not _confirm_run_ok(source_dir):
            return
        select_evaluate_pipeline(source_dir)(**vars(args))
        return

    raise ValueError(
        "This function currently only supports evaluate, export, install, "
        "generate-files, and validate."
    )
