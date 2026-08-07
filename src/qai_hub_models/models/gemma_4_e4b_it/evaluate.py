# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys

from qai_hub_models.models._shared.llm.evaluate import llm_evaluate
from qai_hub_models.models._shared.llm.model import LLM_QNN
from qai_hub_models.models.gemma_4_e4b_it.model import (
    FPSplitModelWrapper,
    Gemma4_E4B_PreSplit,
    Gemma4_E4B_QuantizablePreSplit,
    QuantizedSplitModelWrapper,
)

if __name__ == "__main__":
    use_presplit = "--use-presplit" in sys.argv
    if use_presplit:
        sys.argv.remove("--use-presplit")
    llm_evaluate(
        quantized_model_cls=Gemma4_E4B_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper,
        fp_model_cls=Gemma4_E4B_PreSplit if use_presplit else FPSplitModelWrapper,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
    )
