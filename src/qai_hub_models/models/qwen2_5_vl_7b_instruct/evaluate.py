# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import warnings

from qai_hub_models.models.qwen2_5_vl_7b_instruct.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HF_REPO_NAME,
    Qwen2_5_VL_7B_PreSplit,
    Qwen2_5_VL_7B_QuantizablePreSplit,
    Qwen2_5_VL_7B_VisionEncoder,
)
from qai_hub_models.models.templates.llm.evaluate import (
    build_llm_evaluate_parser,
    llm_evaluate,
)
from qai_hub_models.models.templates.llm.model import LLM_QNN
from qai_hub_models.utils.args import QAIHMArgumentParser


def build_parser() -> QAIHMArgumentParser:
    return build_llm_evaluate_parser(
        quantized_model_cls=Qwen2_5_VL_7B_QuantizablePreSplit,
        fp_model_cls=Qwen2_5_VL_7B_PreSplit,
        vision_encoder_cls=Qwen2_5_VL_7B_VisionEncoder,
    )


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_parser()
    if args is None:
        warnings.warn(
            "Running `python -m qai_hub_models.models.qwen2_5_vl_7b_instruct.evaluate` is "
            "deprecated and will be removed in a future release. "
            "Use `qai-hub-models evaluate qwen2_5_vl_7b_instruct` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args = parser.parse_args()
    llm_evaluate(
        quantized_model_cls=Qwen2_5_VL_7B_QuantizablePreSplit,
        fp_model_cls=Qwen2_5_VL_7B_PreSplit,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
        parser=parser,
        args=args,
        vision_encoder_cls=Qwen2_5_VL_7B_VisionEncoder,
        hf_repo_name=HF_REPO_NAME,
        vlm_image_size=(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
    )


if __name__ == "__main__":
    main()
