# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.templates.wedetect.demo import wedetect_detection_demo
from qai_hub_models.models.wedetect.app import WeDetectModelApp
from qai_hub_models.models.wedetect.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    WeDetectModel,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "room.jpg"
)

DEFAULT_CLASS_LABELS = "bed,vase,shoe,clock"


def main(is_test: bool = False) -> None:
    """Run the WeDetect end-to-end demo."""
    wedetect_detection_demo(
        model_type=WeDetectModel,
        model_id=MODEL_ID,
        image=IMAGE_ADDRESS,
        app_type=WeDetectModelApp,
        class_labels=DEFAULT_CLASS_LABELS,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
