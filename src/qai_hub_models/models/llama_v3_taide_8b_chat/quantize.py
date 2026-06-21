# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.llm.quantize import llm_quantize
from qai_hub_models.models.llama_v3_taide_8b_chat.model import (
    MODEL_ID,
    SUPPORTED_PRECISIONS,
    Llama3_TAIDE_PreSplit,
    Llama3_TAIDE_QuantizablePreSplit,
)

if __name__ == "__main__":
    llm_quantize(
        quantized_model_cls=Llama3_TAIDE_QuantizablePreSplit,
        fp_model_cls=Llama3_TAIDE_PreSplit,
        model_id=MODEL_ID,
        supported_precisions=SUPPORTED_PRECISIONS,
        allow_cpu_to_quantize=True,
    )
