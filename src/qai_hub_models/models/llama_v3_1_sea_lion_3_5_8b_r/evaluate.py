# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys

from qai_hub_models.models._shared.llm.evaluate import llm_evaluate
from qai_hub_models.models._shared.llm.model import LLM_QNN
from qai_hub_models.models.llama_v3_1_sea_lion_3_5_8b_r.model import (
    SUPPORTED_PRECISIONS,
    FPSplitModelWrapper,
    Llama3_1_SEALION_3_5_8B_R_PreSplit,
    Llama3_1_SEALION_3_5_8B_R_QuantizablePreSplit,
    QuantizedSplitModelWrapper,
)

if __name__ == "__main__":
    use_presplit = "--use-presplit" in sys.argv
    llm_evaluate(
        quantized_model_cls=Llama3_1_SEALION_3_5_8B_R_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper,
        fp_model_cls=Llama3_1_SEALION_3_5_8B_R_PreSplit
        if use_presplit
        else FPSplitModelWrapper,
        qnn_model_cls=LLM_QNN,
        supported_precisions=SUPPORTED_PRECISIONS,
    )
