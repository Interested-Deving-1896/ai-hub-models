# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import re

import soundfile
import torch.nn

from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp


def assert_transcription_matches(
    audio_path: str | os.PathLike,
    expected_text: str,
    whisper_encoder: torch.nn.Module,
    whisper_decoder: torch.nn.Module,
    whisper_version: str,
) -> None:
    wav, sample_rate = soundfile.read(audio_path)

    app = HfWhisperApp(whisper_encoder, whisper_decoder, whisper_version)
    transcription = app.transcribe(wav, sample_rate)

    trans = "".join(re.findall(r"\b\w+\b", transcription))
    expected = "".join(re.findall(r"\b\w+\b", expected_text))
    print(
        "\nOriginal_text: ",
        expected_text,
        "\n",
        "Transcription: ",
        transcription,
        sep="",
    )
    assert trans.lower() == expected.lower()
