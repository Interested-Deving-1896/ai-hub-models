# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.openai_clip.app import ClipApp
from qai_hub_models.models.openai_clip.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OpenAIClip,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

DEFAULT_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "image1.jpg"
)
DEFAULT_TEXTS = [
    "a photo of a pyramid in the desert",
    "a photo of a beach at sunset",
    "a photo of a mountain covered in snow",
    "a photo of a city skyline at night",
    "a photo of a forest in autumn",
]


# Run CLIP on a single image with a list of text prompts and return the most relevant.
# Supports both local PyTorch inference and on-device inference via AI Hub.
def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(OpenAIClip)
    parser = get_on_device_demo_parser(parser)
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--texts",
        type=str,
        default="",
        help=f"Comma-separated list of exactly {len(DEFAULT_TEXTS)} text prompts.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    torch_model = model_from_cli_args(OpenAIClip, args)
    model = demo_model_from_cli_args(OpenAIClip, MODEL_ID, args)

    app = ClipApp(model, torch_model.text_tokenizer, torch_model.get_input_spec())  # type: ignore[arg-type]

    image = load_image(args.image)
    texts = [t.strip() for t in args.texts.split(",")] if args.texts else DEFAULT_TEXTS

    similarities = app.predict_best_caption(image, texts).flatten()

    best_idx = int(similarities.argmax())

    for i, (text, score) in enumerate(zip(texts, similarities, strict=False)):
        marker = " <-- best match" if i == best_idx else ""
        print(f"  [{score:.2f}] {text}{marker}")


if __name__ == "__main__":
    main()
