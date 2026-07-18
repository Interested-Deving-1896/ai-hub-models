# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.pipertts.test_utils import (
    pipertts_synthesize_and_verify,
)
from qai_hub_models.models.pipertts_it.demo import main as demo_main
from qai_hub_models.models.pipertts_it.model import PiperTTS_IT
from qai_hub_models.models.whisper_large_v3_turbo.model import WhisperLargeV3Turbo


def test_synthesized_audio() -> None:
    pipertts_synthesize_and_verify(PiperTTS_IT, WhisperLargeV3Turbo)


def test_demo() -> None:
    demo_main(is_test=True)
