# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import pytest

from qai_hub_models.models.qwen2_5_vl_7b_instruct.model import HF_REPO_NAME
from qai_hub_models.models.templates.llm.llm_helpers import create_genie_config
from qai_hub_models.models.templates.qwen2_vl.model import get_vlm_config


@pytest.mark.unmarked
def test_create_genie_config() -> None:
    """The VLM builds a Qwen2.5-VL MRoPE genie config under the
    ``text-generator`` key (not the plain ``dialog`` config the text-only
    LLMs emit). Assert that structure end-to-end.
    """
    context_length = 2048
    # Base transformers can't resolve ``qwen2_5_vl`` via AutoConfig; the model
    # loads it through this helper (which registers the architecture).
    llm_config = get_vlm_config(HF_REPO_NAME)
    text_config = llm_config.text_config
    model_list = [f"qwen2_5_vl_7b_instruct_part_{i}_of_5.bin" for i in range(1, 6)]
    vlm_rope_config: dict[str, Any] = {
        "rope-type": "qwen2vl-mrope",
        "time-step": 50,
        "spatial-merge-size": 2,
        "mrope-section": text_config.rope_scaling["mrope_section"],
    }
    actual_config = create_genie_config(
        context_length=context_length,
        llm_config=text_config,
        embedding_type="rope",
        model_list=model_list,
        embedding_size=text_config.hidden_size,
        top_level_key="text-generator",
        embedding_lut_path="embedding_weights.raw",
        vlm_rope_config=vlm_rope_config,
    )
    kv_dim = text_config.hidden_size // text_config.num_attention_heads
    expected_config: dict[str, Any] = {
        "text-generator": {
            "version": 1,
            "type": "basic",
            "context": {
                "version": 1,
                "size": context_length,
                "n-vocab": text_config.vocab_size,
                "bos-token": text_config.bos_token_id,
                "eos-token": text_config.eos_token_id,
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
                        "kv-dim": kv_dim,
                        "allow-async-init": False,
                        "enable-graph-switching": False,
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
                    "positional-encoding": {
                        "type": "rope",
                        "rope-dim": kv_dim // 2,
                        "rope-theta": int(text_config.rope_theta),
                        "rope-scaling": vlm_rope_config,
                    },
                },
            },
            "embedding": {
                "version": 1,
                "type": "lut",
                "lut-path": "embedding_weights.raw",
                "size": text_config.hidden_size,
                "datatype": "float32",
            },
        }
    }

    assert expected_config == actual_config
