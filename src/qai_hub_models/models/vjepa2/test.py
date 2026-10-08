# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import torch

from qai_hub_models.models.vjepa2.app import VJEPA2App
from qai_hub_models.models.vjepa2.demo import main as demo_main
from qai_hub_models.models.vjepa2.model import FRAMES, HIDDEN, IMG_SIZE, VJEPA2


def test_task() -> None:
    model = VJEPA2.from_pretrained()
    app = VJEPA2App(model)
    clip = torch.rand(3, FRAMES, IMG_SIZE, IMG_SIZE)
    features = app.extract_features(clip)
    assert features.dim() == 3
    assert features.shape[0] == 1
    assert features.shape[-1] == HIDDEN


def test_demo() -> None:
    demo_main(is_test=True)
