# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.imagenet_classifier.demo import imagenet_demo
from qai_hub_models.models.efficientnet_lite4.model import (
    EFFICIENTNET_LITE4_TRANSFORM,
    MODEL_ID,
    EfficientNetLite4,
)


def main(is_test: bool = False) -> None:
    imagenet_demo(
        EfficientNetLite4, MODEL_ID, is_test, transform=EFFICIENTNET_LITE4_TRANSFORM
    )


if __name__ == "__main__":
    main()
