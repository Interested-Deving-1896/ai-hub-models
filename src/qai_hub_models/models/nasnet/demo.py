# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.nasnet.model import (
    MODEL_ID,
    NASNET_TRANSFORM,
    NASNet,
)
from qai_hub_models.models.templates.imagenet_classifier.demo import imagenet_demo


def main(is_test: bool = False) -> None:
    imagenet_demo(NASNet, MODEL_ID, is_test, transform=NASNET_TRANSFORM)


if __name__ == "__main__":
    main()
