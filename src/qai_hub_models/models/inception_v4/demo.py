# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.inception_v4.model import (
    INCEPTION_V4_TRANSFORM,
    MODEL_ID,
    InceptionNetV4,
)
from qai_hub_models.models.templates.imagenet_classifier.demo import imagenet_demo


def main(is_test: bool = False) -> None:
    imagenet_demo(InceptionNetV4, MODEL_ID, is_test, transform=INCEPTION_V4_TRANSFORM)


if __name__ == "__main__":
    main()
