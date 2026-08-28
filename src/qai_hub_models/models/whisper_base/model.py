# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models.templates.hf_whisper.model import (
    TIKTOKEN_URL,
    HfWhisper,
)
from qai_hub_models.models.templates.hf_whisper.utils import (
    write_whisper_supplementary_files,
)
from qai_hub_models.models.templates.hf_whisper.whisper_metadata_json import (
    WhisperCapabilities,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
WHISPER_VERSION = "openai/whisper-base"

WHISPER_BASE_CAPABILITIES = WhisperCapabilities(
    streaming=False,
    file_based=True,
    language_detection=True,
    confidence_scores=False,
)


class WhisperBase(HfWhisper):
    @classmethod
    def get_hf_whisper_version(cls) -> str:
        return WHISPER_VERSION

    def write_supplementary_files(
        self, output_dir: str | os.PathLike, metadata: ModelMetadata
    ) -> None:
        write_whisper_supplementary_files(
            output_dir,
            metadata,
            "whisper-base",
            WHISPER_BASE_CAPABILITIES,
            TIKTOKEN_URL,
            display_name="Whisper Base",
        )
