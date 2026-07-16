# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from enum import Enum
from typing import Literal

from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT, QAIHM_PACKAGE_ROOT

SCORECARD_MODELS_ROOT = QAIHM_PACKAGE_ROOT / "scorecard" / "models"


class TestRunnerSplit(Enum):
    """Named GitHub actions runner splits for grouping models in CI test runs."""

    DEFAULT = "default"
    LLM = "llm"
    PI0_5 = "pi05"

    @property
    def name(self) -> str:
        return self.value

    @property
    def runs_on(self) -> dict[Literal["group", "labels"], str | list[str]] | None:
        """
        Runner configuration for this split, matching GitHub Actions runs-on syntax.

        Returns None (use workflow default) or a dict with "group" and/or "labels"
        (e.g. {"group": "GPU", "labels": ["self-hosted"]}).
        """
        if self is TestRunnerSplit.LLM:
            return {"group": "GPU"}
        return None


class QAIHMModelScorecardConfig(BaseQAIHMConfig):
    """Schema for model scorecard-config.yaml — fields consumed only by internal CI."""

    # If set, skips
    #  - generating `test_generated.py`
    #  - weekly scorecard
    #  - generating perf.yaml
    skip_hub_tests_and_scorecard: bool = False

    # Second knob for skipping of scorecard generation. Use case, skip scorecard but run hub tests.
    skip_scorecard: bool = False

    # If set to true, Scorecard will still run this model, but perf.yaml and associated code-gen.yaml / README.md changes will not be written to disk.
    # This is useful for models whose assets cannot be changed in a release, but we still want to continue testing said models.
    freeze_perf_yaml: bool = False

    # Places this model into a named CI test split.
    test_split: TestRunnerSplit = TestRunnerSplit.DEFAULT

    # If set, disables generating `export.py`.
    skip_export: bool = False

    # If set, disables generating `evaluate.py` (export.py and tests are still generated).
    skip_evaluate: bool = False

    # Overrides the scorecard acceptance threshold. E.g., set this to 15 to
    # allow up to a 15-point drop and still be considered successful.
    numerics_threshold_override: float | None = None

    # When possible, package versions in a model's specific `requirements.txt`
    # should match the versions in `qai_hub_models/global_requirements.txt`.
    # When this is not possible, set this field to indicate an inconsistency.
    global_requirements_incompatible: bool = False

    @classmethod
    def from_model(cls, model_id: str) -> QAIHMModelScorecardConfig:
        """Load scorecard-config.yaml for the given model."""
        if not os.path.exists(QAIHM_MODELS_ROOT / model_id):
            raise ValueError(f"{model_id} does not exist")

        scorecard_path = SCORECARD_MODELS_ROOT / model_id / "scorecard-config.yaml"
        return cls.from_yaml(scorecard_path, create_empty_if_no_file=True)

    @property
    def runs_in_scorecard(self) -> bool:
        """Whether the model runs in scorecard."""
        return not self.skip_hub_tests_and_scorecard and not self.skip_scorecard

    @property
    def is_llm(self) -> bool:
        """
        True if this model is a large language model and produces perf updates
        through a QDC workflow.
        """
        return self.test_split is TestRunnerSplit.LLM
