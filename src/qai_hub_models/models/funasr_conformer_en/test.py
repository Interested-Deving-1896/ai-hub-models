# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import pytest
import soundfile as sf

from qai_hub_models.models.funasr_conformer_en.app import FunASRConformerEnApp
from qai_hub_models.models.funasr_conformer_en.demo import load_demo_audio, main
from qai_hub_models.models.funasr_conformer_en.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FunASRConformerEn,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_raw_file,
)

GROUND_TRUTH_RESULT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "ground_truth.txt"
)


def test_transcribe() -> None:
    model = FunASRConformerEn.from_pretrained()
    app = FunASRConformerEnApp(model, model.frontend, model.token_list)
    wav_file = load_demo_audio()
    audio, sr = sf.read(wav_file, dtype="float32")

    transcription = app.transcribe(audio, sample_rate=sr)

    expected = load_raw_file(GROUND_TRUTH_RESULT).strip()

    assert transcription == expected, "Transcription does not match expected output"


@pytest.mark.trace
def test_trace() -> None:
    model = FunASRConformerEn.from_pretrained()
    traced = model.convert_to_torchscript()
    app = FunASRConformerEnApp(traced, model.frontend, model.token_list)
    wav_file = load_demo_audio()
    audio, sr = sf.read(wav_file, dtype="float32")

    transcription = app.transcribe(audio, sample_rate=sr)

    expected = load_raw_file(GROUND_TRUTH_RESULT).strip()

    assert transcription == expected, "Transcription does not match expected output"


def test_demo() -> None:
    main(is_test=True)
