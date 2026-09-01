# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import qai_hub as hub

from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT
from qai_hub_models.utils.printing import (
    print_file_tree_changes,
    print_on_target_demo_cmd,
)


def test_print_file_tree_changes() -> None:
    out = print_file_tree_changes(
        "/test",
        ["/test/unmodified.txt", "/test/a_test_subdir/unmodified2.txt"],
        ["/test/added.txt", "/test/added_removed.txt"],
        ["/test/removed.txt", "/test/added_removed.txt"],
    )
    ident = "    "
    assert out[1] == "/test"
    assert out[2] == ""
    assert out[3] == f"{ident}a_test_subdir/"
    assert out[4] == f"{ident * 2}unmodified2.txt"
    assert out[5] == ""
    assert out[6] == f"{ident}+ added.txt"
    assert out[7] == f"{ident}-+ added_removed.txt"
    assert out[8] == f"{ident}- removed.txt"
    assert out[9] == f"{ident}unmodified.txt"


def _fake_compile_job(model_id: str) -> MagicMock:
    # spec, or MagicMock's auto __iter__ makes it read as an Iterable of jobs.
    job = MagicMock(spec=hub.CompileJob)
    job.wait.return_value.success = True
    job.get_target_model.return_value.model_id = model_id
    return job


@pytest.mark.parametrize(
    ("folder", "expected_target"),
    [
        (QAIHM_MODELS_ROOT / "resnet50", "resnet50"),
        (Path("/recipes/my_model"), "/recipes/my_model"),
    ],
)
def test_on_target_demo_cmd_uses_the_cli(
    folder: Path, expected_target: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The printed command must be runnable as-is, for in-tree and standalone alike."""
    with patch(
        "qai_hub_models.utils.printing.get_device_and_chipset_name",
        return_value=("Snapdragon 8 Elite QRD", "qualcomm-snapdragon-8-elite"),
    ):
        print_on_target_demo_cmd(
            _fake_compile_job("mabc123"), folder, hub.Device("Snapdragon 8 Elite QRD")
        )

    out = capsys.readouterr().out
    assert f"qai-hub-models demo {expected_target} --eval-mode on-device" in out
    assert "python" not in out
    assert "--hub-model-id mabc123" in out
