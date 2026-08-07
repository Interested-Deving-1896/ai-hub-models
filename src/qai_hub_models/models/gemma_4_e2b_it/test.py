# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for Gemma4-E2B.

The KV-cache layout assertions are shared with E4B; the helpers live in
_shared/gemma4/test_utils.py and are driven here with E2B's geometry (MQA).
"""

from __future__ import annotations

from typing import Any

import pytest
from transformers import AutoConfig

from qai_hub_models.models._shared.gemma4.test_utils import (
    KVLayoutConfig,
    assert_embedding_lut_is_written_blockwise,
    assert_gqa_emits_twice_the_kv_tensors_of_mqa,
    assert_kv_input_tensors_have_batch_dim_one,
    assert_kv_key_value_inner_shapes,
    assert_kv_output_tensors_use_per_head_naming,
    assert_kv_tensor_count_scales_with_num_kv_heads,
)
from qai_hub_models.models._shared.llm.llm_helpers import create_genie_config
from qai_hub_models.models.gemma_4_e2b_it.model import (
    GLOBAL_HEAD_DIM,
    HEAD_DIM,
    HF_REPO_NAME,
    HIDDEN_SIZE,
    NUM_ATTN_HEADS,
    NUM_KEY_VALUE_HEADS,
    NUM_KV_SHARED_LAYERS,
    NUM_LAYERS,
    NUM_SPLITS,
    SLIDING_WINDOW_PATTERN,
)

KV_LAYOUT_CFG = KVLayoutConfig(
    num_hidden_layers=NUM_LAYERS,
    hidden_size=HIDDEN_SIZE,
    num_key_value_heads=NUM_KEY_VALUE_HEADS,
    num_attention_heads=NUM_ATTN_HEADS,
    head_dim=HEAD_DIM,
    global_head_dim=GLOBAL_HEAD_DIM,
    num_kv_shared_layers=NUM_KV_SHARED_LAYERS,
    sliding_window_pattern=SLIDING_WINDOW_PATTERN,
)


def test_kv_input_tensors_have_batch_dim_one() -> None:
    assert_kv_input_tensors_have_batch_dim_one(KV_LAYOUT_CFG)


def test_kv_output_tensors_use_per_head_naming() -> None:
    assert_kv_output_tensors_use_per_head_naming(KV_LAYOUT_CFG)


def test_kv_tensor_count_scales_with_num_kv_heads() -> None:
    assert_kv_tensor_count_scales_with_num_kv_heads(KV_LAYOUT_CFG)


def test_gqa_emits_twice_the_kv_tensors_of_mqa() -> None:
    assert_gqa_emits_twice_the_kv_tensors_of_mqa(KV_LAYOUT_CFG)


def test_kv_key_value_inner_shapes() -> None:
    assert_kv_key_value_inner_shapes(KV_LAYOUT_CFG)


def test_embedding_lut_is_written_blockwise() -> None:
    assert_embedding_lut_is_written_blockwise()


@pytest.mark.unmarked
def test_create_genie_config() -> None:
    context_length = 4096
    # Gemma4Config is a multimodal wrapper; extract the text sub-config.
    llm_config = AutoConfig.from_pretrained(HF_REPO_NAME).get_text_config()
    model_list = [
        f"gemma_4_e2b_it_w4a16_part_{i}_of_{NUM_SPLITS}.bin"
        for i in range(1, NUM_SPLITS + 1)
    ]
    actual_config = create_genie_config(context_length, llm_config, "rope", model_list)
    expected_config: dict[str, Any] = {
        "dialog": {
            "version": 1,
            "type": "basic",
            "context": {
                "version": 1,
                "size": 4096,
                "n-vocab": 262144,
                "bos-token": 2,
                "eos-token": 1,
            },
            "sampler": {
                "version": 1,
                "seed": 42,
                "temp": 0.8,
                "top-k": 40,
                "top-p": 0.95,
            },
            "tokenizer": {"version": 1, "path": "tokenizer.json"},
            "engine": {
                "version": 1,
                "n-threads": 3,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "use-mmap": True,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 0,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": 256,
                        "pos-id-dim": 128,
                        "allow-async-init": False,
                        "rope-theta": 10000,
                    },
                    "extensions": "htp_backend_ext_config.json",
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {
                        "version": 1,
                        "ctx-bins": model_list,
                    },
                },
            },
        }
    }

    assert expected_config == actual_config
