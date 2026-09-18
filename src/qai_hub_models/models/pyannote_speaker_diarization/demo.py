# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import tempfile
import wave

import numpy as np

from qai_hub_models.models.pyannote_speaker_diarization.app import (
    PyannoteSpeakerDiarizationApp,
)
from qai_hub_models.models.pyannote_speaker_diarization.dataset import (
    _AMI_ASSETS,
    _read_wav_mono_16k,
)
from qai_hub_models.models.pyannote_speaker_diarization.model import (
    MODEL_ID,
    SAMPLE_RATE,
    SEG_DURATION_FRAMES,
    PyannoteSpeakerDiarization,
    load_pyannote_pipeline,
)
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)

# Cached 30s demo clip from AMI IS1009a (55-85s).
_DEMO_CLIP_PATH = _AMI_ASSETS[0].path.parent / "demo_audio.wav"


def load_demo_audio() -> str:
    """Return path to the demo audio clip, downloading and extracting if not cached."""
    if _DEMO_CLIP_PATH.exists():
        return str(_DEMO_CLIP_PATH)
    asset = _AMI_ASSETS[0]
    asset.fetch()
    audio = _read_wav_mono_16k(asset.path)
    clip = audio[55 * SAMPLE_RATE : 85 * SAMPLE_RATE]
    _DEMO_CLIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(_DEMO_CLIP_PATH), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((clip * 32767).astype(np.int16).tobytes())
    return str(_DEMO_CLIP_PATH)


def _make_test_audio() -> str:
    """Create a 5-second silent WAV for use in test_demo."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    with wave.open(tmp_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"\x00" * (SEG_DURATION_FRAMES * 2))  # 5 s of silence
    return tmp_path


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(PyannoteSpeakerDiarization)
    parser = get_on_device_demo_parser(parser)
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to the audio file (WAV, 16kHz mono recommended).",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    _, (segmentation, embedding) = demo_model_components_from_cli_args(
        PyannoteSpeakerDiarization, MODEL_ID, args
    )
    app = PyannoteSpeakerDiarizationApp(
        segmentation=segmentation,
        embedding=embedding,
        pipeline=load_pyannote_pipeline(),
    )

    audio_file = args.audio_file or (
        _make_test_audio() if is_test else load_demo_audio()
    )
    diarization = app.diarize(audio_file)
    print("MODEL_ID:", MODEL_ID)
    print("Diarization result:")
    print(diarization)


if __name__ == "__main__":
    main()
