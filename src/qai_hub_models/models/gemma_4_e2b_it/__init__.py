# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Gemma-4-E2B-IT - PreSplit-Part architecture for LLM deployment.

This module provides:
- PreSplit classes (FP and Quantizable) with class-level caching for model + ONNX splitting
- Unified Part classes that handle both FP and Quantizable modes based on precision
- Collection class for deploying the model as a single shard (one context binary)
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
    Gemma4_E2B_Collection,
    Gemma4_E2B_Part1_Of_1,
    Gemma4_E2B_PartBase,
    Gemma4_E2B_PreSplit,
    Gemma4_E2B_QuantizablePreSplit,
    Gemma4_E2B_VisionEncoder,
    QuantizedSplitModelWrapper,
)

Model = Gemma4_E2B_Collection
VisionEncoder = Gemma4_E2B_VisionEncoder

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
    "Gemma4_E2B_Collection",
    "Gemma4_E2B_Part1_Of_1",
    "Gemma4_E2B_PartBase",
    "Gemma4_E2B_PreSplit",
    "Gemma4_E2B_QuantizablePreSplit",
    "Gemma4_E2B_VisionEncoder",
    "Model",
    "QuantizedSplitModelWrapper",
    "SplitForwardMixin",
    "VisionEncoder",
]
