# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys

from qai_hub_models.models.intern3_5_vl_4b.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HF_REPO_NAME,
    FPSplitModelWrapper,
    Intern3_5_VL_4B_PreSplit,
    Intern3_5_VL_4B_QuantizablePreSplit,
    Intern3_5_VL_4B_VisionEncoder,
    QuantizedSplitModelWrapper,
)
from qai_hub_models.models.templates.vlm.evaluate import vlm_evaluate

if __name__ == "__main__":
    use_presplit = "--use-presplit" in sys.argv
    if use_presplit:
        sys.argv.remove("--use-presplit")
    vlm_evaluate(
        quantized_model_cls=Intern3_5_VL_4B_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper,
        fp_model_cls=Intern3_5_VL_4B_PreSplit if use_presplit else FPSplitModelWrapper,
        vision_encoder_cls=Intern3_5_VL_4B_VisionEncoder,
        hf_repo_name=HF_REPO_NAME,
        vlm_image_size=(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
    )
