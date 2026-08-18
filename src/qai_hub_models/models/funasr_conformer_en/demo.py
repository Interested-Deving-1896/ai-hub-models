# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import soundfile as sf

from qai_hub_models.models.funasr_conformer_en.app import FunASRConformerEnApp
from qai_hub_models.models.funasr_conformer_en.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FunASRConformerEn,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset


def load_demo_audio() -> str:
    TEST_AUDIO_PATH = CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "sample.wav"
    ).fetch()
    return str(TEST_AUDIO_PATH)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(FunASRConformerEn)
    parser = get_on_device_demo_parser(parser, add_output_dir=False)
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Audio file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(FunASRConformerEn, MODEL_ID, args)

    local_model = model_from_cli_args(FunASRConformerEn, args)
    app = FunASRConformerEnApp(model, local_model.frontend, local_model.token_list)  # type: ignore[arg-type]

    wav_file = args.audio_file or load_demo_audio()
    audio, sr = sf.read(wav_file, dtype="float32")

    transcription = app.transcribe(audio, sample_rate=sr)

    print(f"Transcription: {transcription}")


if __name__ == "__main__":
    main()
