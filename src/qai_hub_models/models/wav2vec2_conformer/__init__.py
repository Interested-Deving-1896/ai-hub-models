# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.wav2vec2_conformer.app import Wav2Vec2ConformerApp as App
from qai_hub_models.models.wav2vec2_conformer.model import MODEL_ID
from qai_hub_models.models.wav2vec2_conformer.model import (
    Wav2Vec2Conformer as Model,
)

__all__ = ["MODEL_ID", "App", "Model"]
