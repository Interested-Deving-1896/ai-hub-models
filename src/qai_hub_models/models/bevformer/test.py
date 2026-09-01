# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.bevformer.app import BEVFormerApp
from qai_hub_models.models.bevformer.demo import main as demo_main
from qai_hub_models.models.bevformer.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SAMPLE_TOKEN,
    BEVFormer,
    load_frame_inputs,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

GOLDEN_FILENAME = f"output_{SAMPLE_TOKEN}.npz"

GOLDEN_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, GOLDEN_FILENAME
)


def test_task() -> None:
    model = BEVFormer.from_pretrained()
    app = BEVFormerApp(model)

    spec = model.get_input_spec()
    image, can_bus, lidar2img = load_frame_inputs(spec)

    raw = app.predict_3d_boxes(image, can_bus, lidar2img, raw_output=True)
    assert isinstance(raw, tuple), (
        "predict_3d_boxes(raw_output=True) must return a tuple"
    )
    bev_embed, cls_scores, bbox_preds = raw

    golden = np.load(GOLDEN_ASSET.fetch())
    np.testing.assert_allclose(bev_embed, golden["bev_embed"], rtol=1e-03, atol=1e-03)
    np.testing.assert_allclose(cls_scores, golden["cls_scores"], rtol=1e-03, atol=1e-03)
    np.testing.assert_allclose(bbox_preds, golden["bbox_preds"], rtol=1e-03, atol=1e-03)


def test_demo() -> None:
    demo_main(is_test=True)
