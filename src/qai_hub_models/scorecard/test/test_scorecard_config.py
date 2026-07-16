# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest

from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.scorecard.scorecard_config_yaml import QAIHMModelScorecardConfig
from qai_hub_models.utils.path_helpers import MODEL_IDS


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_pip_flags_require_global_requirements_incompatible(model_id: str) -> None:
    """If code-gen.yaml sets pip_install_flags or pip_pre_build_reqs, the model's
    scorecard-config.yaml must have global_requirements_incompatible: true.
    """
    cj = QAIHMModelCodeGen.from_model(model_id)
    if cj.pip_install_flags is None and cj.pip_pre_build_reqs is None:
        return
    sc = QAIHMModelScorecardConfig.from_model(model_id)
    assert sc.global_requirements_incompatible, (
        f"{model_id}: pip_install_flags/pip_pre_build_reqs is set in code-gen.yaml, "
        "but global_requirements_incompatible is not true in scorecard-config.yaml."
    )
