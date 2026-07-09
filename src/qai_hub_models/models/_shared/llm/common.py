# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import gc
import os
from enum import Enum

import torch
from packaging.version import Version

from qai_hub_models.scorecard.device import (
    ScorecardDevice,
    cs_8_elite_qrd,
    cs_x_elite,
)

# Minimum torch version required for dynamic-shape ONNX export (dynamo export).
# Note that earlier versions did support dynamic shapes in general, but did
# not work well for LLMs until 2.10. torch >= 2.11 changes the exported graph
# in ways that break our split LLM pipeline (e.g. llama_v3_2_1b_instruct,
# qwen2_5_vl_7b_instruct), so we pin to 2.10.x.
TORCH_DYNAMIC_SHAPE_MIN_VERSION = "2.10"
TORCH_DYNAMIC_SHAPE_BELOW_VERSION = "2.11"
TORCH_SUPPORTS_DYNAMIC_SHAPE = (
    Version(TORCH_DYNAMIC_SHAPE_MIN_VERSION)
    <= Version(torch.__version__)
    < Version(TORCH_DYNAMIC_SHAPE_BELOW_VERSION)
)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


DEDICATED_POOL_DEVICES: frozenset[ScorecardDevice] = frozenset(
    {cs_8_elite_qrd, cs_x_elite}
)


def get_qdc_api_token(device: ScorecardDevice) -> str:
    """QDC_PRIVATE_API_KEY for the dedicated pool, QDC_API_TOKEN otherwise."""
    if device in DEDICATED_POOL_DEVICES:
        token = os.environ.get("QDC_PRIVATE_API_KEY")
        if not token:
            raise ValueError(
                f"QDC_PRIVATE_API_KEY is not set; required for {device.name} "
                "(dedicated QDC pool)."
            )
        return token
    token = os.environ.get("QDC_API_TOKEN")
    if not token:
        raise ValueError("QDC_API_TOKEN is not set.")
    return token


class LLMIOType(Enum):
    # Genie-compatible input (with input token ids)
    # Inputs:
    # - input_ids (integer token ids)
    # - attention_mask
    # - position_ids_cos (half size)
    # - position_ids_sin (half size)
    genie_input_ids = "genie_input_ids"

    # Genie-compatible input (with input token embeddings)
    # Inputs:
    # - input_embeds (post-Gather token embeddings)
    # - attention_mask
    # - position_ids_cos (half size)
    # - position_ids_sin (half size)
    genie_input_embeds = "genie_inputs_embeds"

    # Hugging Face original input
    # Inputs:
    # - input_ids (integer token ids)
    # - attention_mask
    # - position_ids (integer position ids)
    huggingface_input_ids = "huggingface_input_ids"
