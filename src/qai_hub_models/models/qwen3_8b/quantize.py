# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.qwen3_8b.model import (
    MODEL_ID,
    SUPPORTED_PRECISIONS,
    Qwen3_8B_PreSplit,
    Qwen3_8B_QuantizablePreSplit,
)
from qai_hub_models.models.templates.llm.quantize import llm_quantize

if __name__ == "__main__":
    llm_quantize(
        quantized_model_cls=Qwen3_8B_QuantizablePreSplit,
        fp_model_cls=Qwen3_8B_PreSplit,
        model_id=MODEL_ID,
        supported_precisions=SUPPORTED_PRECISIONS,
    )
