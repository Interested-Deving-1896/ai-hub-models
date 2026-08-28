# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys

from qai_hub_models.models.qwen3_1_7b.model import (
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_1_7B_PreSplit,
    Qwen3_1_7B_QuantizablePreSplit,
)
from qai_hub_models.models.templates.llm.evaluate import llm_evaluate
from qai_hub_models.models.templates.llm.model import LLM_QNN

if __name__ == "__main__":
    use_presplit = "--use-presplit" in sys.argv
    llm_evaluate(
        quantized_model_cls=Qwen3_1_7B_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper,
        fp_model_cls=FPSplitModelWrapper if use_presplit else Qwen3_1_7B_PreSplit,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
    )
