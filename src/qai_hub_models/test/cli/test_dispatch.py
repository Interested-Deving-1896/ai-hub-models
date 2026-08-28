# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for the heavy-side export/evaluate dispatcher."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.cli.dispatch import (
    build_evaluate_parser_for,
    build_export_parser_for,
    run_model_script,
)
from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.utils.base_model import BaseModel


def test_dispatch_export_builds_parser_and_runs() -> None:
    """Export path: parser is built once from resolved recipe and pipeline invoked."""
    fake_parser = argparse.ArgumentParser()
    fake_parser.add_argument("--device")
    fake_parsed = argparse.Namespace(device="S25")

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/fake_model"),
        ) as mock_resolve,
        patch(
            "qai_hub_models.cli.dispatch.build_export_parser_for",
            return_value=fake_parser,
        ) as mock_build,
        patch("qai_hub_models.cli.dispatch.select_pipeline") as mock_select,
        patch("qai_hub_models.cli.dispatch._confirm_run_ok", return_value=True),
    ):
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("fake_model", "export", ["--device", "S25"])

        mock_resolve.assert_called_once_with("fake_model")
        mock_build.assert_called_once()
        mock_select.assert_called_once()


def test_dispatch_export_prompts_for_unpublished_model() -> None:
    """Export for `status: unpublished` recipe prompts and exits early on decline."""
    fake_parser = argparse.ArgumentParser()
    fake_parsed = argparse.Namespace()

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/sam"),
        ),
        patch(
            "qai_hub_models.cli.dispatch.build_export_parser_for",
            return_value=fake_parser,
        ),
        patch("qai_hub_models.cli.dispatch.resolve_manifest") as mock_manifest,
        patch(
            "qai_hub_models.cli.dispatch.check_unpublished_model_warning",
            return_value=False,
        ) as mock_check,
        patch("qai_hub_models.cli.dispatch.select_pipeline") as mock_select,
    ):
        mock_manifest.return_value.status = MODEL_STATUS.UNPUBLISHED
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("sam", "export", [])

        mock_check.assert_called_once()
        mock_select.assert_not_called()


def test_dispatch_export_prompts_for_pending_model() -> None:
    """Export for ``status: pending`` recipe prompts — only PUBLISHED skips."""
    fake_parser = argparse.ArgumentParser()
    fake_parsed = argparse.Namespace()

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/mobile_facenet"),
        ),
        patch(
            "qai_hub_models.cli.dispatch.build_export_parser_for",
            return_value=fake_parser,
        ),
        patch("qai_hub_models.cli.dispatch.resolve_manifest") as mock_manifest,
        patch(
            "qai_hub_models.cli.dispatch.check_unpublished_model_warning",
            return_value=False,
        ) as mock_check,
        patch("qai_hub_models.cli.dispatch.select_pipeline") as mock_select,
    ):
        mock_manifest.return_value.status = MODEL_STATUS.PENDING
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("mobile_facenet", "export", [])

        mock_check.assert_called_once()
        mock_select.assert_not_called()


def test_dispatch_export_does_not_prompt_for_unset_status() -> None:
    """``status: unset`` means external, where the catalog warning does not apply."""
    fake_parser = argparse.ArgumentParser()
    fake_parsed = argparse.Namespace()

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/external_recipe"),
        ),
        patch(
            "qai_hub_models.cli.dispatch.build_export_parser_for",
            return_value=fake_parser,
        ),
        patch("qai_hub_models.cli.dispatch.resolve_manifest") as mock_manifest,
        patch(
            "qai_hub_models.cli.dispatch.check_unpublished_model_warning",
            return_value=False,
        ) as mock_check,
        patch("qai_hub_models.cli.dispatch.select_pipeline") as mock_select,
    ):
        mock_manifest.return_value.status = MODEL_STATUS.UNSET
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("external_recipe", "export", [])

        mock_check.assert_not_called()
        mock_select.assert_called_once()


def test_dispatch_export_skips_prompt_for_published_model() -> None:
    """Export for ``status: published`` recipe runs silently — no prompt."""
    fake_parser = argparse.ArgumentParser()
    fake_parsed = argparse.Namespace()

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/resnet50"),
        ),
        patch(
            "qai_hub_models.cli.dispatch.build_export_parser_for",
            return_value=fake_parser,
        ),
        patch("qai_hub_models.cli.dispatch.resolve_manifest") as mock_manifest,
        patch(
            "qai_hub_models.cli.dispatch.check_unpublished_model_warning",
        ) as mock_check,
        patch("qai_hub_models.cli.dispatch.select_pipeline") as mock_select,
    ):
        mock_manifest.return_value.status = MODEL_STATUS.PUBLISHED
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("resnet50", "export", [])

        mock_check.assert_not_called()
        mock_select.assert_called_once()


def test_dispatch_evaluate_builds_parser_and_runs() -> None:
    """Evaluate path: parser is built once from resolved recipe and pipeline invoked."""
    fake_parser = argparse.ArgumentParser()
    fake_parser.add_argument("--device")
    fake_parsed = argparse.Namespace(device="S25")

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/fake_model"),
        ) as mock_resolve,
        patch(
            "qai_hub_models.cli.dispatch.build_evaluate_parser_for",
            return_value=fake_parser,
        ) as mock_build,
        patch("qai_hub_models.cli.dispatch.select_evaluate_pipeline") as mock_select,
        patch("qai_hub_models.cli.dispatch._confirm_run_ok", return_value=True),
    ):
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("fake_model", "evaluate", ["--device", "S25"])

        mock_resolve.assert_called_once_with("fake_model")
        mock_build.assert_called_once()
        mock_select.assert_called_once()


def test_dispatch_evaluate_prompts_for_unpublished_model() -> None:
    """Evaluate for `status: unpublished` recipe prompts and exits early on decline."""
    fake_parser = argparse.ArgumentParser()
    fake_parsed = argparse.Namespace()

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/sam"),
        ),
        patch(
            "qai_hub_models.cli.dispatch.build_evaluate_parser_for",
            return_value=fake_parser,
        ),
        patch("qai_hub_models.cli.dispatch.resolve_manifest") as mock_manifest,
        patch(
            "qai_hub_models.cli.dispatch.check_unpublished_model_warning",
            return_value=False,
        ) as mock_check,
        patch("qai_hub_models.cli.dispatch.select_evaluate_pipeline") as mock_select,
    ):
        mock_manifest.return_value.status = MODEL_STATUS.UNPUBLISHED
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("sam", "evaluate", [])

        mock_check.assert_called_once()
        mock_select.assert_not_called()


def test_export_parser_uses_export_paths() -> None:
    """
    Bug fix: JIT models were rejecting `--target-runtime qnn_context_binary`
    because the parser was fed `get_supported_paths_for_testing()`, which
    drops AOT runtimes for JIT models. The parser must be fed
    `get_supported_paths_for_export()`.
    """
    manifest = Mock()
    export_paths = {Precision.w8a8: [TargetRuntime.QNN_CONTEXT_BINARY]}
    testing_paths = {Precision.w8a8: [TargetRuntime.QNN_DLC]}
    manifest.get_supported_paths_for_export.return_value = export_paths
    manifest.get_supported_paths_for_testing.return_value = testing_paths
    manifest.default_device = None
    manifest.separate_quantize_script = False

    with (
        patch("qai_hub_models.cli.dispatch.resolve_manifest", return_value=manifest),
        patch("qai_hub_models.cli.dispatch.resolve_model_cls", return_value=Mock()),
        patch("qai_hub_models.cli.dispatch.select_pipeline"),
        patch("qai_hub_models.cli.dispatch.export_parser") as mock_export_parser,
    ):
        build_export_parser_for(Path("/tmp/fake_model"))

        kwargs = mock_export_parser.call_args.kwargs
        assert kwargs["supported_precision_runtimes"] is export_paths
        assert kwargs["supported_precision_runtimes"] is not testing_paths


def test_evaluate_parser_uses_export_paths() -> None:
    """
    Same bug applies to evaluate: JIT models must accept AOT runtimes via
    the JIT-compile + link path. Parser must be fed
    `get_supported_paths_for_export()`.
    """
    manifest = Mock()
    export_paths = {Precision.w8a8: [TargetRuntime.QNN_CONTEXT_BINARY]}
    testing_paths = {Precision.w8a8: [TargetRuntime.QNN_DLC]}
    manifest.get_supported_paths_for_export.return_value = export_paths
    manifest.get_supported_paths_for_testing.return_value = testing_paths
    manifest.default_device = None
    manifest.num_calibration_samples = None
    manifest.can_use_quantize_job = False
    manifest.supports_quantization = False

    model_cls = Mock(spec=BaseModel)
    model_cls.get_eval_dataset_classes.return_value = []

    with (
        patch("qai_hub_models.cli.dispatch.resolve_manifest", return_value=manifest),
        patch("qai_hub_models.cli.dispatch.resolve_model_cls", return_value=model_cls),
        patch("qai_hub_models.cli.dispatch.evaluate_parser") as mock_evaluate_parser,
    ):
        build_evaluate_parser_for(Path("/tmp/fake_model"))

        kwargs = mock_evaluate_parser.call_args.kwargs
        assert kwargs["supported_precision_runtimes"] is export_paths
        assert kwargs["supported_precision_runtimes"] is not testing_paths


class TestDispatchDemo:
    """`demo` hands its argv tail to the recipe's own demo.main()."""

    def _run(self, forwarded: list[str], confirm: bool = True) -> list[str]:
        """Run the demo branch and return the sys.argv its main() observed."""
        seen: list[str] = []
        demo_module = types.ModuleType("fake_model.demo")
        demo_module.main = lambda: seen.extend(sys.argv)  # type: ignore[attr-defined]
        recipe_module = types.ModuleType("fake_model")

        with (
            patch(
                "qai_hub_models.cli.dispatch.resolve_recipe_dir",
                return_value=Path("/tmp/fake_model"),
            ),
            patch(
                "qai_hub_models.cli.dispatch.import_recipe_module",
                return_value=recipe_module,
            ),
            patch(
                "qai_hub_models.cli.dispatch.importlib.import_module",
                return_value=demo_module,
            ),
            patch("qai_hub_models.cli.dispatch._confirm_run_ok", return_value=confirm),
        ):
            run_model_script("fake_model", "demo", forwarded)
        return seen

    def test_forwards_the_tail_as_argv(self) -> None:
        """Demo scripts read sys.argv, so the tail has to arrive that way."""
        assert self._run(["--eval-mode", "on-device"]) == [
            "fake_model.demo",
            "--eval-mode",
            "on-device",
        ]

    def test_restores_argv_afterwards(self) -> None:
        before = list(sys.argv)
        self._run(["--image", "x.png"])
        assert sys.argv == before

    def test_declining_the_prompt_skips_the_demo(self) -> None:
        assert self._run([], confirm=False) == []
