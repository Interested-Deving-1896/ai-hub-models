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
import importlib
import sys
from pathlib import Path
from typing import cast

from qai_hub_models.cli.generate_files import main as generate_files_main
from qai_hub_models.cli.install import main as install_main
from qai_hub_models.cli.upload_to_hf import main as upload_to_hf_main
from qai_hub_models.cli.validate import main as validate_main
from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.utils.args import evaluate_parser, export_parser
from qai_hub_models.utils.asset_loaders import check_unpublished_model_warning
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate.dispatch import select_evaluate_pipeline
from qai_hub_models.utils.export.context import (
    import_recipe_module,
    resolve_manifest,
    resolve_model_cls,
    resolve_recipe_dir,
)
from qai_hub_models.utils.export.dispatch import select_pipeline

_PROMPT_STATUSES = (MODEL_STATUS.UNPUBLISHED, MODEL_STATUS.PENDING)


def _confirm_run_ok(source_dir: Path) -> bool:
    """Return False iff the user declines the unpublished-recipe warning.

    Only ``unpublished`` and ``pending`` prompt. Both are in-tree statuses that
    say something true: the recipe is in the Qualcomm catalog but has not
    cleared review, so the warning's offer of no support is accurate.

    ``unset`` does not prompt. It is the default every external / standalone
    recipe ships with, and in-tree recipes must set an explicit status
    (enforced in ``test_manifest_yamls.py``), so ``unset`` means "authored
    outside the catalog" — where the warning is simply wrong. It told authors
    their own recipe might not meet standards Qualcomm never applied to it, on
    every single run. The generated card's banner already skips external
    recipes for the same reason.
    """
    manifest = resolve_manifest(source_dir)
    if manifest.status in _PROMPT_STATUSES:
        return check_unpublished_model_warning()
    return True


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
        A recipe target — either a folder path (``my_model``, ``Path``, or
        path string) or a bare installed model id (``"yolov8_det"``).
        Kept named ``model_id`` for call-site compatibility. The lean CLI
        rejects ``script="install"``/``"generate-files"``/``"validate"``
        with a folder before this is called.
    script
        Script name: ``"demo"``, ``"export"``, ``"evaluate"``, ``"install"``,
        ``"generate-files"``, ``"validate"``, or ``"upload-to-hf"``.
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

    if script == "upload-to-hf":
        upload_to_hf_main([str(model_id), *forwarded])
        return

    source_dir = resolve_recipe_dir(model_id)
    if script == "demo":
        demo_module = importlib.import_module(
            f"{import_recipe_module(source_dir).__name__}.demo"
        )
        if not _confirm_run_ok(source_dir):
            return
        # Demo scripts build their own parser internally and read sys.argv
        # rather than taking argv, so the tail is handed over that way.
        saved_argv = sys.argv
        sys.argv = [f"{source_dir.name}.demo", *forwarded]
        try:
            demo_module.main()
        finally:
            sys.argv = saved_argv
        return

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
        "This function currently only supports demo, evaluate, export, install, "
        "generate-files, validate, and upload-to-hf."
    )
