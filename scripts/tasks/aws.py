# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from pathlib import Path

from .constants import REPO_ROOT
from .task import ConditionalTask, NoOpTask, RunCommandsWithVenvTask
from .util import on_ci


class ValidateAwsCredentialsTask(ConditionalTask):
    def __init__(self, venv_path: str | None) -> None:
        aws_script_path = Path(REPO_ROOT, "scripts", "aws").as_posix()
        install_saml2aws_path = f"{aws_script_path}/install_saml2aws.sh"
        super().__init__(
            group_name=None,
            condition=on_ci,
            true_task=NoOpTask("Not validating AWS credentials on CI"),
            false_task=RunCommandsWithVenvTask(
                "Validating AWS credentials",
                venv_path,
                [
                    f'if [ -d "{aws_script_path}" ]; then bash "{install_saml2aws_path}"; fi',
                    "qai-hub-models validate_aws_credentials",
                ],
            ),
        )
