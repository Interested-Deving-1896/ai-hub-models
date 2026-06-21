# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.llm.model import SplitForwardMixin

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
    Falcon3_7B_Collection,
    Falcon3_7B_Part1_Of_5,
    Falcon3_7B_Part2_Of_5,
    Falcon3_7B_Part3_Of_5,
    Falcon3_7B_Part4_Of_5,
    Falcon3_7B_Part5_Of_5,
    Falcon3_7B_PartBase,
    Falcon3_7B_PreSplit,
    Falcon3_7B_QuantizablePreSplit,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
)

Model = Falcon3_7B_Collection

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
    "Falcon3_7B_Collection",
    "Falcon3_7B_Part1_Of_5",
    "Falcon3_7B_Part2_Of_5",
    "Falcon3_7B_Part3_Of_5",
    "Falcon3_7B_Part4_Of_5",
    "Falcon3_7B_Part5_Of_5",
    "Falcon3_7B_PartBase",
    "Falcon3_7B_PreSplit",
    "Falcon3_7B_QuantizablePreSplit",
    "Model",
    "QuantizedSplitModelWrapper",
    "SplitForwardMixin",
]
