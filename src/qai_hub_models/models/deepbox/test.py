# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.deepbox.app import DeepBoxApp
from qai_hub_models.models.deepbox.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.deepbox.demo import main as demo_main
from qai_hub_models.models.deepbox.model import MODEL_ASSET_VERSION, MODEL_ID, DeepBox
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output_image_yolov3u.png"
)


def test_task() -> None:
    image = load_image(
        INPUT_IMAGE_ADDRESS,
    )
    expected_output = load_image(
        OUTPUT_IMAGE_ADDRESS,
    )
    wrapper = DeepBox.from_pretrained()
    app = DeepBoxApp.from_pretrained(wrapper)
    assert np.allclose(np.asarray(app.detect_image(image)), np.asarray(expected_output))


def test_demo() -> None:
    demo_main(is_test=True)
