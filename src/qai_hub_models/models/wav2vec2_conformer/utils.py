# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Any

import numpy as np
from transformers import Wav2Vec2Processor


def normalize_audio(
    processor: Wav2Vec2Processor,
    audio: np.ndarray,
    sample_rate: int,
    max_length: int = 160000,
) -> Any:
    """Canonical processor call used by both App and Dataset."""
    return processor(
        audio,
        sampling_rate=sample_rate,
        max_length=max_length,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
