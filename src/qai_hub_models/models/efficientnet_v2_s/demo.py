# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.efficientnet_v2_s.model import (
    EFFICIENTNET_V2_S_TRANSFORM,
    MODEL_ID,
    EfficientNetV2s,
)
from qai_hub_models.models.templates.imagenet_classifier.demo import imagenet_demo


def main(is_test: bool = False) -> None:
    imagenet_demo(
        EfficientNetV2s, MODEL_ID, is_test, transform=EFFICIENTNET_V2_S_TRANSFORM
    )


if __name__ == "__main__":
    main()
