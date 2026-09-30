# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL import ImageDraw

from qai_hub_models.models.superpoint.app import SuperPointApp
from qai_hub_models.models.superpoint.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SuperPoint,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "superpoint.jpg"
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(SuperPoint)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path or URL of the input image. Defaults to the bundled test image.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(SuperPoint, MODEL_ID, args)
    app = SuperPointApp(model)  # type: ignore[arg-type]

    image = load_image(args.image if args.image else IMAGE_ADDRESS)
    keypoints, scores, _ = app.predict(image)

    # Overlay keypoints on the original image
    vis = image.convert("RGB")
    draw = ImageDraw.Draw(vis)
    radius = 3
    for (x, y), s in zip(keypoints.tolist(), scores.tolist(), strict=False):
        color = (int(255 * s), 255 - int(255 * s), 0)
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            outline=color,
            width=1,
        )

    if not is_test:
        print(f"Detected {len(keypoints)} keypoints.")
        display_or_save_image(
            vis, args.output_dir, "superpoint_keypoints.png", MODEL_ID
        )


if __name__ == "__main__":
    main()
