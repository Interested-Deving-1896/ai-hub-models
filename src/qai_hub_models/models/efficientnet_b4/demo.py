# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.efficientnet_b4.model import (
    EFFICIENTNET_B4_TRANSFORM,
    MODEL_ID,
    EfficientNetB4,
)
from qai_hub_models.models.templates.imagenet_classifier.demo import imagenet_demo


def main(is_test: bool = False) -> None:
    imagenet_demo(
        EfficientNetB4, MODEL_ID, is_test, transform=EFFICIENTNET_B4_TRANSFORM
    )


if __name__ == "__main__":
    main()
