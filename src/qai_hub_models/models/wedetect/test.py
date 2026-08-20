# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import pytest
import torch

from qai_hub_models.models._shared.wedetect.app import tokenize_class_names
from qai_hub_models.models.wedetect.app import WeDetectModelApp
from qai_hub_models.models.wedetect.demo import main as demo_main
from qai_hub_models.models.wedetect.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    WeDetectModel,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "room.jpg"
)

CLASS_LABELS = ["bed"]
EXPECTED_BOX = torch.tensor([166.24, 362.51, 418.65, 481.15])
EXPECTED_SCORE = 0.41


def test_task() -> None:
    """Verify that WeDetect detects the bed with reasonable box + score."""
    image = load_image(INPUT_IMAGE_ADDRESS)
    collection = WeDetectModel.from_pretrained()
    app = WeDetectModelApp.from_components(list(collection.components.values()))

    boxes, scores, class_idx = app.predict_boxes_from_image(
        image, class_labels=CLASS_LABELS, raw_output=True
    )
    assert len(boxes) == 1 and boxes[0].shape == (1, 4)
    assert torch.allclose(boxes[0][0], EXPECTED_BOX, atol=1.0)
    assert abs(float(scores[0][0]) - EXPECTED_SCORE) < 0.02
    assert int(class_idx[0][0]) == 0


@pytest.mark.trace
def test_trace() -> None:
    """Verify that torch.export.export of the detector produces numerically consistent output."""
    collection = WeDetectModel.from_pretrained()
    detector = collection.detector
    text_encoder = collection.text_encoder

    input_spec = detector.get_input_spec()
    sample_inputs = detector.sample_inputs(input_spec)
    image = torch.from_numpy(sample_inputs["image"][0])
    txt_feats = torch.from_numpy(sample_inputs["txt_feats"][0])

    eager_out = detector(image, txt_feats)

    exported = torch.export.export(detector, (image, txt_feats))

    exported_out = exported.module()(image, txt_feats)

    for eager_t, exported_t in zip(eager_out, exported_out, strict=False):
        assert torch.allclose(eager_t, exported_t, atol=1e-4)

    # Verify text encoder produces expected shape for a small prompt.
    input_ids, attention_mask = tokenize_class_names(
        text_encoder.tokenizer,
        ["bed", "vase", "clock"],
        num_classes=3,
        max_seq_len=text_encoder.max_seq_len,
    )
    feats = text_encoder(input_ids, attention_mask)
    assert feats.shape == (3, 768)


def test_demo() -> None:
    """Run demo and verify it does not crash."""
    demo_main(is_test=True)
