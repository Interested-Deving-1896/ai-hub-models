# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.silero_vad.app import SileroVADApp
from qai_hub_models.models.silero_vad.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SileroVAD,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

# A short earnings-call audio clip from Earnings-21 eval-10 (Zagg Inc, Q3).
# Sourced from: https://github.com/revdotcom/speech-datasets/tree/d5d09d014bb2fd2e8837ba097b75527174677ff1/earnings21
DEMO_AUDIO_ASSET = CachedWebModelAsset(
    "https://raw.githubusercontent.com/revdotcom/speech-datasets/d5d09d014bb2fd2e8837ba097b75527174677ff1/earnings21/media/4387332.mp3",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "4387332.mp3",
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(SileroVAD)
    parser = get_on_device_demo_parser(parser, add_output_dir=False)
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to an audio file to run VAD on. Defaults to a sample from Earnings-21.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    audio_path = args.audio
    if audio_path is None:
        audio_path = str(DEMO_AUDIO_ASSET.fetch())

    print(f"Running Silero VAD on: {audio_path}")
    model = demo_model_from_cli_args(SileroVAD, MODEL_ID, args)
    app = SileroVADApp(model)
    segments = app.predict(audio_path)

    if not segments:
        print("No speech detected.")
        return

    print(f"Detected {len(segments)} speech segment(s):")
    for i, seg in enumerate(segments, 1):
        duration = seg["end"] - seg["start"]
        print(
            f"  [{i:3d}]  {seg['start']:8.3f}s  ->  {seg['end']:8.3f}s  ({duration:.3f}s)"
        )


if __name__ == "__main__":
    main()
