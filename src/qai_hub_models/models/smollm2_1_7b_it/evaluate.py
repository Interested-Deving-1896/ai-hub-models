# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import warnings

from qai_hub_models.models.smollm2_1_7b_it.model import (
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Smollm2_1_7B_Instruct_PreSplit,
    Smollm2_1_7B_Instruct_QuantizablePreSplit,
)
from qai_hub_models.models.templates.llm.evaluate import (
    build_llm_evaluate_parser,
    llm_evaluate,
)
from qai_hub_models.models.templates.llm.model import LLM_QNN
from qai_hub_models.utils.args import QAIHMArgumentParser


def build_parser() -> QAIHMArgumentParser:
    return build_llm_evaluate_parser(
        quantized_model_cls=QuantizedSplitModelWrapper,
        fp_model_cls=Smollm2_1_7B_Instruct_PreSplit,
        has_presplit=True,
    )


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_parser()
    if args is None:
        warnings.warn(
            "Running `python -m qai_hub_models.models.smollm2_1_7b_it.evaluate` is "
            "deprecated and will be removed in a future release. "
            "Use `qai-hub-models evaluate smollm2_1_7b_it` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args = parser.parse_args()
    use_presplit = args.use_presplit
    llm_evaluate(
        quantized_model_cls=Smollm2_1_7B_Instruct_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper,
        fp_model_cls=FPSplitModelWrapper
        if use_presplit
        else Smollm2_1_7B_Instruct_PreSplit,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
        parser=parser,
        args=args,
    )


if __name__ == "__main__":
    main()
