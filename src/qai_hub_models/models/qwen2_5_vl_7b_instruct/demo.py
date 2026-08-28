# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.qwen2_5_vl_7b_instruct import MODEL_ID
from qai_hub_models.models.qwen2_5_vl_7b_instruct.model import (
    HF_REPO_NAME,
    HF_REPO_URL,
    Qwen2_5_VL_7B_PreSplit,
    Qwen2_5_VL_7B_QuantizablePreSplit,
    Qwen2_5_VL_7B_VisionEncoder,
)
from qai_hub_models.models.templates.llm.demo import llm_chat_demo
from qai_hub_models.models.templates.llm.model import LLM_QNN
from qai_hub_models.models.templates.qwen2_vl.model import (
    DEFAULT_USER_PROMPT,
    END_TOKENS,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec


def qwen2_5_vl_7b_instruct_chat_demo(
    test_checkpoint: CheckpointSpec | None = None,
) -> None:
    llm_chat_demo(
        model_cls=Qwen2_5_VL_7B_QuantizablePreSplit,
        fp_model_cls=Qwen2_5_VL_7B_PreSplit,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
        model_id=MODEL_ID,
        end_tokens=END_TOKENS,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_url=HF_REPO_URL,
        default_prompt=DEFAULT_USER_PROMPT,
        test_checkpoint=test_checkpoint,
        vision_encoder_cls=Qwen2_5_VL_7B_VisionEncoder,
    )


def main() -> None:
    qwen2_5_vl_7b_instruct_chat_demo()


if __name__ == "__main__":
    main()
