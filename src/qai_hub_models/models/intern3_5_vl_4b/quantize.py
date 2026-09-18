# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.intern3_5_vl_4b.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    MODEL_ID,
    SAMPLE_IMAGE,
    SUPPORTED_PRECISIONS,
    Intern3_5_VL_4B_PreSplit,
    Intern3_5_VL_4B_QuantizablePreSplit,
    Intern3_5_VL_4B_VisionEncoder,
)
from qai_hub_models.models.templates.vlm.quantize import quantize_vlm


def main() -> None:
    quantize_vlm(
        quantized_model_cls=Intern3_5_VL_4B_QuantizablePreSplit,
        fp_model_cls=Intern3_5_VL_4B_PreSplit,
        vision_encoder_cls=Intern3_5_VL_4B_VisionEncoder,
        supported_precisions=SUPPORTED_PRECISIONS,
        description="Quantize InternVL3.5-4B model",
        model_id=MODEL_ID,
        sample_image=SAMPLE_IMAGE,
        default_image_height=DEFAULT_IMAGE_HEIGHT,
        default_image_width=DEFAULT_IMAGE_WIDTH,
    )


if __name__ == "__main__":
    main()
