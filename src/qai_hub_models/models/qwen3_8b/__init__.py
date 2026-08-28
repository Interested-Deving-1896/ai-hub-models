# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Qwen3-8B - PreSplit-Part architecture for LLM deployment.

This module provides:
- PreSplit classes (FP and Quantizable) with class-level caching for model + ONNX splitting
- Unified Part classes that handle both FP and Quantizable modes based on precision
- Collection class for deploying the model as 5 splits
"""

from qai_hub_models.models.templates.llm.model import SplitForwardMixin

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
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_8B_Collection,
    Qwen3_8B_Part1_Of_5,
    Qwen3_8B_Part2_Of_5,
    Qwen3_8B_Part3_Of_5,
    Qwen3_8B_Part4_Of_5,
    Qwen3_8B_Part5_Of_5,
    Qwen3_8B_PartBase,
    Qwen3_8B_PreSplit,
    Qwen3_8B_QuantizablePreSplit,
)

Model = Qwen3_8B_Collection

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
    "FPSplitModelWrapper",
    "Model",
    "QuantizedSplitModelWrapper",
    "Qwen3_8B_Collection",
    "Qwen3_8B_Part1_Of_5",
    "Qwen3_8B_Part2_Of_5",
    "Qwen3_8B_Part3_Of_5",
    "Qwen3_8B_Part4_Of_5",
    "Qwen3_8B_Part5_Of_5",
    "Qwen3_8B_PartBase",
    "Qwen3_8B_PreSplit",
    "Qwen3_8B_QuantizablePreSplit",
    "SplitForwardMixin",
]
