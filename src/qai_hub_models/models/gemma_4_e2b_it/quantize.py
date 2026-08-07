# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
r"""
Quantization script for Gemma4-E2B.

Delegates to the shared Gemma4 two-stage workflow. See
``qai_hub_models.models._shared.gemma4.quantize`` for the full description.

Usage
-----
Stage A (CPU ok):
    python -m qai_hub_models.models.gemma_4_e2b_it.quantize \
        --export-only --checkpoint <checkpoint-dir> -o onnx_export/

Stage B (GPU):
    python -m qai_hub_models.models.gemma_4_e2b_it.quantize \
        --checkpoint <checkpoint-dir> --onnx-dir onnx_export/ -o checkpoint/w4a16/ \
        --num-samples 20

Single-stage (export then calibrate in one run):
    python -m qai_hub_models.models.gemma_4_e2b_it.quantize \
        --checkpoint <checkpoint-dir> -o checkpoint/w4a16/ --num-samples 20

GEMMA4_LOCAL_CHECKPOINT can supply the checkpoint dir instead of --checkpoint.
"""

from __future__ import annotations

from qai_hub_models.models._shared.gemma4 import quantize as _shared_quantize
from qai_hub_models.models.gemma_4_e2b_it.model import (
    MODEL_ID,
    SUPPORTED_PRECISIONS,
    Gemma4_E2B_PreSplit,
    Gemma4_E2B_QuantizablePreSplit,
)

export_onnx = _shared_quantize.export_onnx
quantize = _shared_quantize.quantize


def main() -> None:
    _shared_quantize.main(
        presplit_cls=Gemma4_E2B_PreSplit,
        quant_presplit_cls=Gemma4_E2B_QuantizablePreSplit,
        model_id=MODEL_ID,
        supported_precisions=SUPPORTED_PRECISIONS,
    )


if __name__ == "__main__":
    main()
