# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import soundfile as sf

from qai_hub_models.utils.asset_loaders import PathType, load_raw_file


class ConformerApp(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


def conformer_transcribe_test_e2e(
    app: ConformerApp,
    load_demo_audio: Callable[[], str],
    ground_truth_result: PathType,
) -> None:
    wav_file = load_demo_audio()
    audio, sr = sf.read(wav_file, dtype="float32")

    transcription = app.transcribe(audio, sample_rate=sr)

    expected = load_raw_file(ground_truth_result).strip()
    assert transcription == expected, "Transcription does not match expected output"
