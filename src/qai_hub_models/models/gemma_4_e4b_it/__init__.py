# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Gemma-4-E4B-IT - PreSplit-Part architecture for LLM deployment.

This module provides:
- PreSplit classes (FP and Quantizable) with class-level caching for model + ONNX splitting
- Unified Part classes that handle both FP and Quantizable modes based on precision
- Collection class for deploying the model as 4 splits
"""

from qai_hub_models.models._shared.llm.model import SplitForwardMixin

from .model import (
    DEFAULT_PRECISION,
    GLOBAL_HEAD_DIM,
    HEAD_DIM,
    HF_REPO_NAME,
    HIDDEN_SIZE,
    MIN_MEMORY_RECOMMENDED,
    MODEL_ID,
    NUM_ATTN_HEADS,
    NUM_KEY_VALUE_HEADS,
    NUM_KV_SHARED_LAYERS,
    NUM_LAYERS,
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
    FPSplitModelWrapper,
    Gemma4_E4B_Collection,
    Gemma4_E4B_Part1_Of_4,
    Gemma4_E4B_Part2_Of_4,
    Gemma4_E4B_Part3_Of_4,
    Gemma4_E4B_Part4_Of_4,
    Gemma4_E4B_PartBase,
    Gemma4_E4B_PreSplit,
    Gemma4_E4B_QuantizablePreSplit,
    Gemma4_E4B_VisionEncoder,
    QuantizedSplitModelWrapper,
)

Model = Gemma4_E4B_Collection
VisionEncoder = Gemma4_E4B_VisionEncoder

__all__ = [
    "DEFAULT_PRECISION",
    "GLOBAL_HEAD_DIM",
    "HEAD_DIM",
    "HF_REPO_NAME",
    "HIDDEN_SIZE",
    "MIN_MEMORY_RECOMMENDED",
    "MODEL_ID",
    "NUM_ATTN_HEADS",
    "NUM_KEY_VALUE_HEADS",
    "NUM_KV_SHARED_LAYERS",
    "NUM_LAYERS",
    "NUM_LAYERS_PER_SPLIT",
    "NUM_SPLITS",
    "FPSplitModelWrapper",
    "Gemma4_E4B_Collection",
    "Gemma4_E4B_Part1_Of_4",
    "Gemma4_E4B_Part2_Of_4",
    "Gemma4_E4B_Part3_Of_4",
    "Gemma4_E4B_Part4_Of_4",
    "Gemma4_E4B_PartBase",
    "Gemma4_E4B_PreSplit",
    "Gemma4_E4B_QuantizablePreSplit",
    "Gemma4_E4B_VisionEncoder",
    "Model",
    "QuantizedSplitModelWrapper",
    "SplitForwardMixin",
    "VisionEncoder",
]
