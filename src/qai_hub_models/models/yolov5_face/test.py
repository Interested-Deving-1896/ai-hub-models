# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.models.yolov5_face.app import YoloV5FaceApp
from qai_hub_models.models.yolov5_face.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov5_face.demo import main as demo_main
from qai_hub_models.models.yolov5_face.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV5Face,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

OUTPUT_GOLDEN_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolov5_face_golden.npz"
).fetch()


def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    model = YoloV5Face.from_pretrained()
    app = YoloV5FaceApp(model, input_spec=model.get_input_spec())

    (boxes,), (scores,), (landmarks,) = app.run_inference_on_image(
        image, raw_output=True
    )
    with np.load(OUTPUT_GOLDEN_ADDRESS) as data:
        np.testing.assert_allclose(boxes.numpy(), data["boxes"], rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(scores.numpy(), data["scores"], rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(
            landmarks.numpy(), data["landmarks"], rtol=1e-3, atol=1e-3
        )


def test_demo() -> None:
    demo_main(is_test=True)
