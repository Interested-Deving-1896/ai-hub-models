# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.resnet101.model import MODEL_ID, ResNet101
from qai_hub_models.models.templates.imagenet_classifier.demo import imagenet_demo


def main(is_test: bool = False) -> None:
    imagenet_demo(ResNet101, MODEL_ID, is_test)


if __name__ == "__main__":
    main()
