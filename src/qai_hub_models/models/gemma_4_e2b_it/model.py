# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Gemma4 E2B IT - PreSplit-Part architecture for LLM deployment.

Model: google/gemma-4-E2B-it
Architecture: Gemma4 with sliding window + global attention, PLE, MQA
"""

from __future__ import annotations

import logging

from qai_hub_models import Precision
from qai_hub_models.models._shared.gemma4.model import (
    Gemma4PartBase,
    Gemma4PreSplitBase,
    Gemma4PreSplitCollectionBase,
    Gemma4QuantizablePreSplitBase,
)
from qai_hub_models.models._shared.gemma4.vision_encoder import Gemma4VisionEncoder
from qai_hub_models.models._shared.llm.common import LLMIOType  # noqa: F401
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_CONTEXT_LENGTHS as GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS as GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS,
)
from qai_hub_models.models._shared.llm.model import SplitForwardMixin

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_CONTEXT_LENGTHS = GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Model architecture constants (from Gemma4-E2B config)
NUM_LAYERS = 35
NUM_KV_SHARED_LAYERS = 20  # Layers 15-34 share KV
NUM_KV_LAYERS = NUM_LAYERS - NUM_KV_SHARED_LAYERS  # 15 layers with own KV

# 1-shard: the whole decoder + lm_head compile to ONE context binary. The
# generic geniex-qairt-plugin runtime infers the shard count from the
# ``_S_of_T`` suffix in the graph names, so a single shard emits
# ``prompt_ar128_cl4096_1_of_1`` / ``token_ar1_cl4096_1_of_1`` and one .bin.
#   Part 1: layers 0-34 + lm_head (whole model)
# ``split_lm_head=False`` keeps the LM head fused into the single part;
# ``splitting_points=None`` (no internal cuts) so ``split_onnx`` returns the
# whole model as one bundle.
NUM_SPLITS = 1
SPLIT_LM_HEAD = False
SPLITTING_POINTS = None
NUM_LAYERS_PER_SPLIT = None  # unused for a single split

HIDDEN_SIZE = 1536
NUM_KEY_VALUE_HEADS = 1  # MQA
NUM_ATTN_HEADS = 8
HEAD_DIM = 256  # SWA layers
GLOBAL_HEAD_DIM = 512  # Full attention layers
SLIDING_WINDOW_PATTERN = 5  # 4 SWA + 1 global
SLIDING_WINDOW = 512  # SWA KV window size

HF_REPO_NAME = "google/gemma-4-E2B-it"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 20

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
# precision -> published asset name (fetched by from_pretrained("DEFAULT_W4A16")).
DEFAULT_CHECKPOINT = {
    Precision.w4a16: "gemma_4_e2b_it_w4a16",
}

# Name used for split ONNX file basenames
SPLIT_MODEL_NAME = "Gemma4_E2B"

# Vision encoder (VEG) default trace resolution. The Gemma4 processor applies
# pan-and-scan tiling, so the actual patch count is derived from this at load
# time (see Gemma4VisionEncoder.from_pretrained).
VISION_DEFAULT_IMAGE_SIZE = 448


class Gemma4_E2B_VisionEncoder(Gemma4VisionEncoder):
    """Gemma4-E2B Visual Embedding Generator (VEG), exported as FP16."""

    _hf_repo_name = HF_REPO_NAME
    default_image_size = VISION_DEFAULT_IMAGE_SIZE


class Gemma4_E2B_PreSplit(Gemma4PreSplitBase):
    """FP PreSplit for Gemma4-E2B."""

    num_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    global_head_dim = GLOBAL_HEAD_DIM
    num_kv_shared_layers = NUM_KV_SHARED_LAYERS
    sliding_window_pattern = SLIDING_WINDOW_PATTERN
    sliding_window = SLIDING_WINDOW
    hf_repo_name = HF_REPO_NAME

    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    split_lm_head = SPLIT_LM_HEAD
    splitting_points = SPLITTING_POINTS

    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION


class Gemma4_E2B_QuantizablePreSplit(
    Gemma4QuantizablePreSplitBase[Gemma4_E2B_PreSplit]
):
    """Quantizable PreSplit for Gemma4-E2B."""

    FPModel = Gemma4_E2B_PreSplit

    num_layers = NUM_LAYERS
    num_kv_shared_layers = NUM_KV_SHARED_LAYERS
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    supported_precisions = SUPPORTED_PRECISIONS
    default_precision = DEFAULT_PRECISION

    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    split_lm_head = SPLIT_LM_HEAD
    splitting_points = SPLITTING_POINTS


class Gemma4_E2B_PartBase(Gemma4PartBase):
    """Part base for Gemma4-E2B."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    global_head_dim = GLOBAL_HEAD_DIM
    num_layers = NUM_LAYERS
    num_kv_shared_layers = NUM_KV_SHARED_LAYERS
    sliding_window_pattern = SLIDING_WINDOW_PATTERN
    sliding_window = SLIDING_WINDOW
    default_precision = DEFAULT_PRECISION
    fp_presplit_cls = Gemma4_E2B_PreSplit
    quant_presplit_cls = Gemma4_E2B_QuantizablePreSplit


class Gemma4_E2B_Part1_Of_1(Gemma4_E2B_PartBase):
    """The single 1-shard Part: whole decoder + lm_head."""

    part_id = 1


_SPLIT_PART_CLASSES: list[type] = [
    Gemma4_E2B_Part1_Of_1,
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Gemma4_E2B_QuantizablePreSplit
):
    """Quantized eval via split Parts."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Gemma4_E2B_PreSplit):  # type: ignore[misc]
    """FP eval via split Parts."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class Gemma4_E2B_Collection(Gemma4PreSplitCollectionBase):
    """1-shard Collection for Gemma4-E2B (whole model in one context binary)."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Gemma4_E2B_PreSplit
    part_base_cls = Gemma4_E2B_PartBase
    vision_encoder_cls = Gemma4_E2B_VisionEncoder
    parts = {
        "part1_of_1": Gemma4_E2B_Part1_Of_1,
    }
