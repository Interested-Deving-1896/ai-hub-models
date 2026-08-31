# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import sys
import types
from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock, patch

import pytest

from qai_hub_models_cli.cli import (
    _check_version_match,
    main,
)


def _subcommand_choices(parser: argparse.ArgumentParser) -> set[str]:
    """Return the set of subcommand names registered on the top-level parser."""
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    return set(subparsers_action.choices)


def _stub_heavy_modules(
    model_ids: set[str],
    run_model_script: MagicMock | None = None,
    static_help: bool = False,
) -> dict[str, types.ModuleType]:
    """Patch the heavy ``qai_hub_models`` modules the dispatch code imports.

    ``static_help`` is what ``print_command_help`` returns: False stands for a
    command whose flags come from the resolved recipe, so the lean CLI prints
    its own hint instead.
    """
    path_helpers = types.ModuleType("qai_hub_models.utils.path_helpers")
    path_helpers.MODEL_IDS = model_ids  # type: ignore[attr-defined]
    dispatch = types.ModuleType("qai_hub_models.cli.dispatch")
    dispatch.run_model_script = run_model_script or MagicMock()  # type: ignore[attr-defined]
    command_help = types.ModuleType("qai_hub_models.cli.command_help")
    command_help.print_command_help = lambda script, stream: static_help  # type: ignore[attr-defined]
    return {
        "qai_hub_models": types.ModuleType("qai_hub_models"),
        "qai_hub_models.utils": types.ModuleType("qai_hub_models.utils"),
        "qai_hub_models.utils.path_helpers": path_helpers,
        "qai_hub_models.cli": types.ModuleType("qai_hub_models.cli"),
        "qai_hub_models.cli.dispatch": dispatch,
        "qai_hub_models.cli.command_help": command_help,
    }


@pytest.mark.parametrize(
    ("cli_v", "models_v", "should_exit"),
    [
        ("1.0.0", "1.0.0", False),  # match
        ("1.0.0", "2.0.0", True),  # mismatch
        ("1.0.0", None, False),  # qai_hub_models not installed
    ],
)
def test_check_version_match(
    cli_v: str, models_v: str | None, should_exit: bool
) -> None:
    """`_check_version_match` exits iff cli/models versions are both installed and differ."""

    def _version(pkg: str) -> str:
        if pkg == "qai_hub_models_cli":
            return cli_v
        if models_v is None:
            raise PackageNotFoundError(pkg)
        return models_v

    with patch("qai_hub_models_cli.cli.version", side_effect=_version):
        if should_exit:
            with pytest.raises(SystemExit):
                _check_version_match()
        else:
            _check_version_match()


# ── two-phase dispatch for export/evaluate/install ──────────────────


@pytest.mark.parametrize("script", ["export", "evaluate", "install"])
def test_dispatch_forwards_remaining_args_to_model_parser(script: str) -> None:
    """`<script> <model> --flag value` hands model-specific args to dispatch verbatim.

    When the model arg is already a valid installed model ID, dispatch skips the
    (slow) manifest lookup and forwards straight to the model's parser.
    """
    fake_entry = MagicMock()
    fake_entry.id = "mobilenet_v2"
    mock_run = MagicMock()
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch("qai_hub_models_cli.cli.CURRENT_VERSION", "9.9.9"),
        patch(
            "qai_hub_models_cli.cli.get_manifest_entry", return_value=fake_entry
        ) as mock_get_entry,
        patch.dict(
            sys.modules,
            _stub_heavy_modules({"mobilenet_v2"}, run_model_script=mock_run),
        ),
    ):
        main([script, "mobilenet_v2", "--target-runtime", "tflite"])
    # mobilenet_v2 is a valid installed model ID, so the manifest lookup is skipped.
    mock_get_entry.assert_not_called()
    mock_run.assert_called_once_with(
        model_id="mobilenet_v2",
        script=script,
        forwarded=["--target-runtime", "tflite"],
    )


def test_dispatch_missing_model_arg_exits_with_usage_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`export` (no target) prints our usage hint as a usage error.

    Stderr and exit 2, matching argparse, since a missing argument is an error
    rather than a help request.
    """
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch.dict(sys.modules, _stub_heavy_modules(set())),
        pytest.raises(SystemExit) as exc_info,
    ):
        main(["export"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "Usage:" in err
    assert "export <target>" in err


def test_dispatch_help_flag_exits_zero_on_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`export -h` is a help request, so stdout and exit 0."""
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch.dict(sys.modules, _stub_heavy_modules(set())),
        pytest.raises(SystemExit) as exc_info,
    ):
        main(["export", "-h"])
    assert exc_info.value.code == 0
    assert "export <target>" in capsys.readouterr().out


def test_dispatch_builtin_id_wins_over_alias() -> None:
    """A registered alias must not shadow a built-in model id.

    ``register_alias``'s collision check only fires when heavy is installed,
    so someone could register e.g. ``mobilenet_v2`` from a lean CLI and later
    install heavy — dispatch has to prefer the built-in either way.
    """
    mock_run = MagicMock()
    mock_resolve_alias = MagicMock(return_value=None)
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch("qai_hub_models_cli.cli.resolve_alias", mock_resolve_alias),
        patch.dict(
            sys.modules,
            _stub_heavy_modules({"mobilenet_v2"}, run_model_script=mock_run),
        ),
    ):
        main(["export", "mobilenet_v2"])
    mock_resolve_alias.assert_not_called()
    mock_run.assert_called_once_with(
        model_id="mobilenet_v2", script="export", forwarded=[]
    )
