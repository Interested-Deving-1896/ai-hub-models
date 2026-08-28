# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.owl_vit.model import (
    HF_MODEL_ID,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OwlViT,
)
from qai_hub_models.models.templates.owl.demo import owl_detection_demo
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

DEFAULT_TEXT_QUERIES = ["a photo of a cup"]
IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/input_image.jpg"
)


def main(is_test: bool = False) -> None:
    owl_detection_demo(
        OwlViT,
        MODEL_ID,
        HF_MODEL_ID,
        IMAGE_ADDRESS,
        DEFAULT_TEXT_QUERIES,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
