# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torchvision.models as tv_models
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "IMAGENET1K_V1"


class MobileNetV3Small(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> Self:
        net = tv_models.mobilenet_v3_small(weights=weights)
        return cls(net)

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """Lite-MP: promote the top 1% most-sensitive layers to int16."""
        return 1
