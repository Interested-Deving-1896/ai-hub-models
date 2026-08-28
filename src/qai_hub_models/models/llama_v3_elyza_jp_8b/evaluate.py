# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys

from qai_hub_models.models.llama_v3_elyza_jp_8b.model import (
    FPSplitModelWrapper,
    Llama3_Elyza_JP_8B_PreSplit,
    Llama3_Elyza_JP_8B_QuantizablePreSplit,
    QuantizedSplitModelWrapper,
)
from qai_hub_models.models.templates.llm.evaluate import llm_evaluate
from qai_hub_models.models.templates.llm.model import LLM_QNN

if __name__ == "__main__":
    use_presplit = "--use-presplit" in sys.argv
    llm_evaluate(
        quantized_model_cls=Llama3_Elyza_JP_8B_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper,
        fp_model_cls=Llama3_Elyza_JP_8B_PreSplit
        if use_presplit
        else FPSplitModelWrapper,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
    )
