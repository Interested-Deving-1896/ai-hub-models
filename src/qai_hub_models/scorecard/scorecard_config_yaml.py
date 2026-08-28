# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from enum import Enum, unique
from typing import Literal

from pydantic import Field, model_validator

from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    QAIHM_MODELS_ROOT,
    QAIHM_PACKAGE_ROOT,
)

SCORECARD_MODELS_ROOT = QAIHM_PACKAGE_ROOT / "scorecard" / "models"


@unique
class LLMWeekendGroup(Enum):
    WEEK1 = "week1"
    WEEK2 = "week2"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


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

    @property
    def max_models_per_split(self) -> int:
        if self is TestRunnerSplit.LLM:
            return 5
        return 10**9


class QAIHMModelScorecardConfig(BaseQAIHMConfig):
    """Schema for model scorecard-config.yaml — fields consumed only by internal CI."""

    # If set, skips
    #  - generating `test_generated.py`
    #  - weekly scorecard
    #  - generating perf.yaml
    skip_hub_tests_and_scorecard: bool = False

    # Second knob for skipping of scorecard generation. Use case, skip scorecard but run hub tests.
    skip_scorecard: bool = False

    # If set to true, Scorecard will still run this model, but perf.yaml and associated manifest.yaml / README.md changes will not be written to disk.
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

    # Weekend LLM scorecard rotation bucket. Required for scorecard-eligible LLMs.
    weekend_group: LLMWeekendGroup | None = None

    # True if the LLM publishes downloadable release assets (rerun on a QAIRT bump).
    downloadable_llm_asset: bool = False

    # Components benchmarked standalone, mapped to their model card tab label (also the
    # component's perf.yaml key). Unlisted components are LLM backbone parts, measured
    # together end-to-end and consolidated into one tab keyed by the model name.
    standalone_components: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_standalone_component_labels(self) -> QAIHMModelScorecardConfig:
        # Labels are perf.yaml component keys, so a duplicate silently merges two
        # components' metrics.
        labels = list(self.standalone_components.values())
        if len(labels) != len(set(labels)):
            raise ValueError(
                f"standalone_components labels must be unique, got {sorted(labels)}."
            )
        return self

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


def _scorecard_llm_configs() -> dict[str, QAIHMModelScorecardConfig]:
    """{model_id: scorecard_config} for every scorecard-eligible test_split: llm model."""
    out: dict[str, QAIHMModelScorecardConfig] = {}
    for model_id in MODEL_IDS:
        sc = QAIHMModelScorecardConfig.from_model(model_id)
        if sc.is_llm and sc.runs_in_scorecard:
            out[model_id] = sc
    return out


def get_llm_model_ids() -> set[str]:
    """Scorecard-eligible pytorch recipes with test_split == llm."""
    return set(_scorecard_llm_configs())


def get_week_model_ids(week: LLMWeekendGroup) -> set[str]:
    """LLM model IDs assigned to the given weekend rotation bucket."""
    return {m for m, sc in _scorecard_llm_configs().items() if sc.weekend_group is week}


def get_downloadable_llm_model_ids() -> set[str]:
    """LLM model IDs that publish downloadable release assets."""
    return {
        m for m, sc in _scorecard_llm_configs().items() if sc.downloadable_llm_asset
    }


def validate_llm_weekend_coverage() -> None:
    """Every scorecard-eligible test_split: llm model must set weekend_group. Raises on drift."""
    missing = sorted(
        m for m, sc in _scorecard_llm_configs().items() if sc.weekend_group is None
    )
    if missing:
        raise ValueError(
            "Scorecard-eligible test_split: llm models are missing weekend_group in "
            f"scorecard-config.yaml (must be week1 or week2): {missing}."
        )
