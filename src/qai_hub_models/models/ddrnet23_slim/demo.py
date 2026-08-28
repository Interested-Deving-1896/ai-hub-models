# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.ddrnet23_slim.model import (
    INPUT_IMAGE_ADDRESS,
    MODEL_ID,
    DDRNet,
)
from qai_hub_models.models.templates.segmentation.demo import segmentation_demo


def main(is_test: bool = False) -> None:
    segmentation_demo(DDRNet, MODEL_ID, INPUT_IMAGE_ADDRESS, is_test)


if __name__ == "__main__":
    main()
