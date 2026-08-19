# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models.siglip2.app import SigLIP2App
from qai_hub_models.models.siglip2.demo import main as demo_main
from qai_hub_models.models.siglip2.model import MODEL_ASSET_VERSION, MODEL_ID, SigLIP2
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

IMAGE_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "image1.jpg"
)
TEXTS = ["a pyramid in the desert", "a dog on a beach", "a cat on a sofa"]
# "a pyramid in the desert" should score highest for image1.jpg
EXPECTED_BEST_IDX = 0


def test_task() -> None:
    model = SigLIP2.from_pretrained()
    app = SigLIP2App(
        image_encoder=model.image_encoder,
        text_encoder=model.text_encoder,
        logit_scale=model.logit_scale,
        logit_bias=model.logit_bias,
    )
    image = load_image(IMAGE_ASSET)
    logits = app.predict_similarity([image], TEXTS)
    assert logits.shape == (1, len(TEXTS))
    best = int(torch.argmax(logits, dim=-1).item())
    assert best == EXPECTED_BEST_IDX, (
        f"Expected best match index {EXPECTED_BEST_IDX}, got {best}. Logits: {logits}"
    )


def test_demo() -> None:
    demo_main(is_test=True)
