# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL.Image import Image

from qai_hub_models.models.superpoint.app import SuperPointApp
from qai_hub_models.models.superpoint.model import (
    INPUT_IMAGE_ADDRESS_1,
    INPUT_IMAGE_ADDRESS_2,
    MODEL_ID,
    SuperPoint,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.display import display_or_save_image


def main(is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(SuperPoint)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image1",
        type=str,
        default=INPUT_IMAGE_ADDRESS_1,
        help="first image file path or URL",
    )
    parser.add_argument(
        "--image2",
        type=str,
        default=INPUT_IMAGE_ADDRESS_2,
        help="second image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    # Get model input dimensions
    input_spec = SuperPoint.from_pretrained().get_input_spec()
    (_, _, _, height, width) = input_spec["image"][0]

    # Load images — SuperPointApp handles resize and grayscale conversion internally
    image1 = load_image(args.image1)
    image2 = load_image(args.image2)

    model = demo_model_from_cli_args(SuperPoint, MODEL_ID, args)
    print("Model Loaded")

    app = SuperPointApp(
        model,  # type: ignore[arg-type]
        height=height,
        width=width,
    )

    output = app.predict(image1, image2)
    assert isinstance(output, Image)

    if not is_test:
        display_or_save_image(
            output,
            args.output_dir,
            "superpoint_matches.png",
        )


if __name__ == "__main__":
    main()
