# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.templates.yolo.demo import yolo_detection_demo
from qai_hub_models.models.yolov4.app import YoloV4TinyDetectionApp
from qai_hub_models.models.yolov4.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV4Tiny,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/input_image.jpg"
)


def main(is_test: bool = False) -> None:
    yolo_detection_demo(
        YoloV4Tiny,
        MODEL_ID,
        YoloV4TinyDetectionApp,
        IMAGE_ADDRESS,
        YoloV4Tiny.STRIDE_MULTIPLE,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
