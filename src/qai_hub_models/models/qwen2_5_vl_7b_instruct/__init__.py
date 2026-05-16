# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Qwen2.5-VL 7B - PreSplit-Part architecture for VLM deployment.

This module provides:
- PreSplit classes (FP and Quantizable) with class-level caching
- Unified Part classes for split inference
- Vision encoder for on-device export (FP + quantized)
- Collection class for deploying as 5 text splits + 1 vision encoder
"""

from qai_hub_models.models._shared.qwen2_vl.model import (
    Qwen2VLPositionProcessor as PositionProcessor,
)

from .model import (
    DEFAULT_PRECISION,
    HF_REPO_NAME,
    HIDDEN_SIZE,
    MIN_MEMORY_RECOMMENDED,
    MODEL_ID,
    NUM_ATTN_HEADS,
    NUM_KEY_VALUE_HEADS,
    NUM_LAYERS,
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
    Qwen2_5_VL_7B_Collection,
    Qwen2_5_VL_7B_Part1_Of_5,
    Qwen2_5_VL_7B_Part2_Of_5,
    Qwen2_5_VL_7B_Part3_Of_5,
    Qwen2_5_VL_7B_Part4_Of_5,
    Qwen2_5_VL_7B_Part5_Of_5,
    Qwen2_5_VL_7B_PartBase,
    Qwen2_5_VL_7B_PreSplit,
    Qwen2_5_VL_7B_QuantizablePreSplit,
    Qwen2_5_VL_7B_VisionEncoder,
)

VisionEncoder = Qwen2_5_VL_7B_VisionEncoder
Model = Qwen2_5_VL_7B_Collection

__all__ = [
    "DEFAULT_PRECISION",
    "HF_REPO_NAME",
    "HIDDEN_SIZE",
    "MIN_MEMORY_RECOMMENDED",
    "MODEL_ID",
    "NUM_ATTN_HEADS",
    "NUM_KEY_VALUE_HEADS",
    "NUM_LAYERS",
    "NUM_LAYERS_PER_SPLIT",
    "NUM_SPLITS",
    "Model",
    "PositionProcessor",
    "Qwen2_5_VL_7B_Collection",
    "Qwen2_5_VL_7B_Part1_Of_5",
    "Qwen2_5_VL_7B_Part2_Of_5",
    "Qwen2_5_VL_7B_Part3_Of_5",
    "Qwen2_5_VL_7B_Part4_Of_5",
    "Qwen2_5_VL_7B_Part5_Of_5",
    "Qwen2_5_VL_7B_PartBase",
    "Qwen2_5_VL_7B_PreSplit",
    "Qwen2_5_VL_7B_QuantizablePreSplit",
    "VisionEncoder",
]
