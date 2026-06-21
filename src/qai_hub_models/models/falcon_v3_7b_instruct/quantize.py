# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.llm.quantize import llm_quantize
from qai_hub_models.models.falcon_v3_7b_instruct.model import (
    MODEL_ID,
    SUPPORTED_PRECISIONS,
    Falcon3_7B_PreSplit,
    Falcon3_7B_QuantizablePreSplit,
)

if __name__ == "__main__":
    llm_quantize(
        quantized_model_cls=Falcon3_7B_QuantizablePreSplit,
        fp_model_cls=Falcon3_7B_PreSplit,
        model_id=MODEL_ID,
        supported_precisions=SUPPORTED_PRECISIONS,
        allow_cpu_to_quantize=True,
    )
