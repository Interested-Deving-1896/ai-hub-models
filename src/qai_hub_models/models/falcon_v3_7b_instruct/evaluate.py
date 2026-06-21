# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys

from qai_hub_models.models._shared.llm.evaluate import llm_evaluate
from qai_hub_models.models._shared.llm.model import LLM_QNN
from qai_hub_models.models.falcon_v3_7b_instruct.model import (
    SUPPORTED_PRECISIONS,
    Falcon3_7B_PreSplit,
    Falcon3_7B_QuantizablePreSplit,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
)

if __name__ == "__main__":
    use_presplit = "--use-presplit" in sys.argv
    llm_evaluate(
        quantized_model_cls=Falcon3_7B_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper,
        fp_model_cls=Falcon3_7B_PreSplit if use_presplit else FPSplitModelWrapper,
        qnn_model_cls=LLM_QNN,
        supported_precisions=SUPPORTED_PRECISIONS,
    )
