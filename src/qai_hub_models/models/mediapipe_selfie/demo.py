# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.mediapipe_selfie.model import (
    IMAGE_ADDRESS,
    MODEL_ID,
    MediapipeSelfie,
)
from qai_hub_models.models.templates.selfie_segmentation.demo import (
    selfie_segmentation_demo,
)


def main(is_test: bool = False) -> None:
    selfie_segmentation_demo(MediapipeSelfie, MODEL_ID, IMAGE_ADDRESS, is_test)


if __name__ == "__main__":
    main()
