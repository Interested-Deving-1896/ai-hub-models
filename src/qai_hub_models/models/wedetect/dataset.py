# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.templates.wedetect.dataset import WeDetectCocoDataset
from qai_hub_models.models.wedetect.external_repos import EXTERNAL_REPO_PATHS


class WeDetectMainCocoDataset(WeDetectCocoDataset):
    """WeDetect-flavoured COCO detection dataset for the base/tiny models.

    Uses XLM-RoBERTa base tokenizer shipped alongside the wedetect repo.
    """

    @classmethod
    def tokenizer_path(cls) -> str:
        return str(EXTERNAL_REPO_PATHS["wedetect"] / "xlm-roberta-base")
