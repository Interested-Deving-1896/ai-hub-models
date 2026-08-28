# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.efficientvit_l2_cls.model import (
    EFFICIENTVIT_L2_TRANSFORM,
    MODEL_ID,
    EfficientViT,
)
from qai_hub_models.models.templates.imagenet_classifier.demo import imagenet_demo


def main(is_test: bool = False) -> None:
    imagenet_demo(EfficientViT, MODEL_ID, is_test, transform=EFFICIENTVIT_L2_TRANSFORM)


if __name__ == "__main__":
    main()
