# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.imagenet_classifier.demo import imagenet_demo
from qai_hub_models.models.repvit.model import (
    MODEL_ID,
    REPVIT_TRANSFORM,
    RepViT,
)


def main(is_test: bool = False) -> None:
    imagenet_demo(RepViT, MODEL_ID, is_test, transform=REPVIT_TRANSFORM)


if __name__ == "__main__":
    main()
