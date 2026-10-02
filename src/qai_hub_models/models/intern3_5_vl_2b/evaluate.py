# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import warnings

from qai_hub_models.models.intern3_5_vl_2b.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HF_REPO_NAME,
    FPSplitModelWrapper,
    Intern3_5_VL_2B_PreSplit,
    Intern3_5_VL_2B_QuantizablePreSplit,
    Intern3_5_VL_2B_VisionEncoder,
    QuantizedSplitModelWrapper,
)
from qai_hub_models.models.templates.llm.evaluate import (
    build_llm_evaluate_parser,
)
from qai_hub_models.models.templates.vlm.evaluate import vlm_evaluate
from qai_hub_models.utils.args import QAIHMArgumentParser


def build_parser() -> QAIHMArgumentParser:
    return build_llm_evaluate_parser(
        quantized_model_cls=QuantizedSplitModelWrapper,
        fp_model_cls=FPSplitModelWrapper,
        vision_encoder_cls=Intern3_5_VL_2B_VisionEncoder,
        has_presplit=True,
    )


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_parser()
    if args is None:
        warnings.warn(
            "Running `python -m qai_hub_models.models.intern3_5_vl_2b.evaluate` is "
            "deprecated and will be removed in a future release. "
            "Use `qai-hub-models evaluate intern3_5_vl_2b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args = parser.parse_args()
    use_presplit = args.use_presplit
    vlm_evaluate(
        quantized_model_cls=Intern3_5_VL_2B_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper,
        fp_model_cls=Intern3_5_VL_2B_PreSplit if use_presplit else FPSplitModelWrapper,
        parser=parser,
        args=args,
        vision_encoder_cls=Intern3_5_VL_2B_VisionEncoder,
        hf_repo_name=HF_REPO_NAME,
        vlm_image_size=(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
    )


if __name__ == "__main__":
    main()
