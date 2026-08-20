# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from qai_hub_models.models._shared.wedetect.app import WeDetectApp
from qai_hub_models.models.wedetect.external_repos import EXTERNAL_REPO_PATHS


class WeDetectModelApp(WeDetectApp):
    """WeDetect app for the base/tiny models with a bundled XLM-R tokenizer."""

    @classmethod
    def _load_tokenizer(cls) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(
            str(EXTERNAL_REPO_PATHS["wedetect"] / "xlm-roberta-base")
        )
