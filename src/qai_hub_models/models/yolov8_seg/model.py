# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

from typing_extensions import Self
from ultralytics.models import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models import Precision
from qai_hub_models.models.templates.ultralytics.segmentation_model import (
    UltralyticsMulticlassSegmentor,
)
from qai_hub_models.models.templates.yolo.model import YoloSegEvalMixin

MODEL_ASSET_VERSION = 2
MODEL_ID = __name__.split(".")[-2]

SUPPORTED_WEIGHTS = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
]
DEFAULT_WEIGHTS = "yolov8n-seg.pt"


class YoloV8Segmentor(UltralyticsMulticlassSegmentor, YoloSegEvalMixin):
    @classmethod
    def from_pretrained(
        cls, ckpt_name: str = DEFAULT_WEIGHTS, precision: Precision | None = None
    ) -> Self:
        if ckpt_name not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported checkpoint name provided {ckpt_name}.\n"
                f"Supported checkpoints are {list(SUPPORTED_WEIGHTS)}."
            )
        model = cast(SegmentationModel, ultralytics_YOLO(ckpt_name).model)
        return cls(model, precision)

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        # tf_enhanced (the AUTO default at w8a8) clips the boundary outliers mask
        # mAP depends on, costing 1.2 mAP versus min_max.
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        return options + " --range_scheme min_max"
