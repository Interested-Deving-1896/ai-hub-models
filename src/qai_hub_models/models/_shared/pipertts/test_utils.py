# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qai_hub_models.models._shared.hf_whisper.model import HfWhisper
from qai_hub_models.models._shared.pipertts.app import DEFAULT_TEXTS, PiperTTSApp
from qai_hub_models.models._shared.voiceai_tts.test_utils import (
    assert_transcription_matches,
)

if TYPE_CHECKING:
    from qai_hub_models.models._shared.pipertts.model import PiperTTS


def pipertts_synthesize_and_verify(
    model_cls: type[PiperTTS],
    whisper_model_cls: type[HfWhisper],
) -> None:
    # Pin Flow's randn-initialized fixed_noise so Whisper round-trip is stable.
    torch.manual_seed(42)
    model = model_cls.from_pretrained()
    language = model.get_language()
    out_audio_path = PiperTTSApp(
        model.encoder, model.sdp, model.flow, model.decoder, language
    ).predict(DEFAULT_TEXTS[language])

    whisper_model = whisper_model_cls.from_pretrained()
    assert_transcription_matches(
        out_audio_path,
        DEFAULT_TEXTS[language],
        whisper_model.encoder,
        whisper_model.decoder,
        whisper_model_cls.get_hf_whisper_version(),
    )
