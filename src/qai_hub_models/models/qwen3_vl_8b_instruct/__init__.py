# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.templates.qwen3_vl.model import (
    Qwen3VLPositionProcessor as PositionProcessor,
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
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_VL_8B_Collection,
    Qwen3_VL_8B_Part1_Of_6,
    Qwen3_VL_8B_Part2_Of_6,
    Qwen3_VL_8B_Part3_Of_6,
    Qwen3_VL_8B_Part4_Of_6,
    Qwen3_VL_8B_Part5_Of_6,
    Qwen3_VL_8B_Part6_Of_6,
    Qwen3_VL_8B_PartBase,
    Qwen3_VL_8B_PreSplit,
    Qwen3_VL_8B_QuantizablePreSplit,
    Qwen3_VL_8B_VisionEncoder,
)

VisionEncoder = Qwen3_VL_8B_VisionEncoder
Model = Qwen3_VL_8B_Collection

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
    "PositionProcessor",
    "QuantizedSplitModelWrapper",
    "Qwen3_VL_8B_Collection",
    "Qwen3_VL_8B_Part1_Of_6",
    "Qwen3_VL_8B_Part2_Of_6",
    "Qwen3_VL_8B_Part3_Of_6",
    "Qwen3_VL_8B_Part4_Of_6",
    "Qwen3_VL_8B_Part5_Of_6",
    "Qwen3_VL_8B_Part6_Of_6",
    "Qwen3_VL_8B_PartBase",
    "Qwen3_VL_8B_PreSplit",
    "Qwen3_VL_8B_QuantizablePreSplit",
    "VisionEncoder",
]
