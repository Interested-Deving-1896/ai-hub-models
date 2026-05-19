# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.yolo.demo import yolo_obb_demo
from qai_hub_models.models.yolov8_obb.app import YoloV8OBBApp
from qai_hub_models.models.yolov8_obb.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV8OBB,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/input_image.png"
)


def main(is_test: bool = False) -> None:
    yolo_obb_demo(
        YoloV8OBB,
        MODEL_ID,
        YoloV8OBBApp,
        IMAGE_ADDRESS,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
